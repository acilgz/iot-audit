from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from data_loading import _read_csv, sha256
from scaler_regression_test import validated_fit_indices

def group_codes(df, columns):
    if not columns:
        raise ValueError('ERROR: Empty columns')
    elif len(set(columns)) != len(columns):
        raise ValueError('ERROR: Duplicate columns')
    return df.groupby(columns, dropna=False, sort=False, observed=True).ngroup().to_numpy(dtype=np.int64)

def duplicate_counts(codes):
    _, counts = np.unique(codes, return_counts=True)
    return {
        'rows': int(len(codes)), 'distinct_groups': int(len(counts)),
        'duplicate_groups': int(np.count_nonzero(counts > 1)),
        'rows_in_duplicate_groups': int(counts[counts > 1].sum()),
        'duplicate_rows_beyond_first': int(len(codes) - len(counts)),
    }

def overlap(codes, left, right, targets):
    left_codes, right_codes = codes[left], codes[right]
    shared = np.intersect1d(left_codes, right_codes)
    left_matches, right_matches = np.isin(left_codes, shared), np.isin(right_codes, shared)
    per_class = []
    for label in np.unique(targets[right]):
        mask = targets[right] == label
        support = int(mask.sum())
        matched = int(np.count_nonzero(mask & right_matches))
        per_class.append({'target_value': str(label), 'right_rows': support,
                          'right_rows_with_left_match': matched,
                          'fraction': matched / support})
    return {
        'shared_distinct_groups': int(len(shared)),
        'left_rows': int(len(left)), 'right_rows': int(len(right)),
        'left_rows_with_right_match': int(left_matches.sum()),
        'right_rows_with_left_match': int(right_matches.sum()),
        'right_fraction_with_left_match': float(right_matches.mean()),
        'right_overlap_by_class': per_class,
    }

def conflicting_groups(codes, target):
    pairs = pd.DataFrame({'group': codes, 'target': target}).drop_duplicates()
    sizes = pairs.groupby('group', sort=False).size()
    conflict_ids = sizes.index[sizes > 1].to_numpy()
    return {'groups_with_multiple_target_values': int(len(conflict_ids)),
            'rows_in_conflicting_groups': int(np.isin(codes, conflict_ids).sum())}

def git_record():
    result = {'commit': None, 'status_porcelain': None}
    try:
        repo = Path(__file__).resolve().parent.parent
        for key, arguments in [('commit', ['rev-parse', 'HEAD']),
                               ('status_porcelain', ['status', '--porcelain=v1', '--untracked-files=all'])]:
            result[key] = subprocess.run(['git', '-C', str(repo), *arguments],
                                        check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        result['unavailable_reason'] = str(exc)
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--csv', type=Path, default=Path('data/train_test_network.csv'))
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f'ERROR: {args.output} already exists')
    started = datetime.now(timezone.utc).isoformat()
    dataset_hash = sha256(args.csv)
    df = _read_csv(str(args.csv))
    target_names = {'label', 'type', 'target'}
    predictors = [c for c in df.columns if str(c).lower() not in target_names]
    cache = {}
    result = {
        'definition': 'Exact equality of parsed values on the listed columns',
        'scope': 'Read-only audit of existing splits',
        'limitation': 'Selected raw inputs are measured before imputation/OHE',
        'dataset': {'path': str(args.csv.resolve()), 'sha256': dataset_hash,
                    'rows': len(df), 'columns': len(df.columns)},
        'tasks': {},
        'provenance': {'started_at_utc': started, 'command_argv': [sys.executable, *sys.argv],
                       'cwd': str(Path.cwd()), 'run_dir': str(args.run_dir.resolve()),
                       'git': git_record(), 'platform': platform.platform(),
                       'python': platform.python_version(), 'pandas': pd.__version__,
                       'numpy': np.__version__, 'script_sha256': sha256(Path(__file__))},
    }
    for task, model, target_column in [('binary', 'mlp', 'label'), ('multiclass', 'mlp_mc', 'type')]:
        meta_path = args.run_dir / task / 'models' / model / 'preprocessor_meta.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['dataset_sha256'] != dataset_hash or meta['model_name'] != model or meta['mode'] != task:
            raise ValueError(f'{task}: dataset or metadata identity mismatch')
        fit = validated_fit_indices(meta, len(df))
        partitions = {'train': np.asarray(meta['train_indices'], dtype=np.int64),
                      'test': np.asarray(meta['test_indices'], dtype=np.int64),
                      'fit': fit,
                      'validation': np.asarray(meta['internal_split']['validation_indices'], dtype=np.int64)}
        selected = list(dict.fromkeys(meta['numeric_features'] + meta['categorical_features']))
        if any(str(c).lower() in target_names for c in selected):
            raise ValueError(f'{task}: target column in selected inputs')
        targets = df[target_column].astype(str).str.strip().to_numpy()
        task_result = {
            'model_feature_contract': model, 'target_column': target_column,
            'metadata': {'path': str(meta_path.resolve()), 'sha256': sha256(meta_path),
                         'training_commit_as_recorded': meta.get('commit')},
            'split_sizes': {k: len(v) for k, v in partitions.items()},
            'projections': {},
        }
        for name, columns in [('full_rows_including_targets', list(df.columns)),
                              ('raw_predictors_without_targets', predictors),
                              ('selected_raw_model_inputs', selected)]:
            key = tuple(columns)
            if key not in cache:
                cache[key] = group_codes(df, columns)
            codes = cache[key]
            projection = {'columns': columns, 'global': duplicate_counts(codes),
                          'within_partitions': {k: duplicate_counts(codes[v]) for k, v in partitions.items()},
                          'target_conflicts': conflicting_groups(codes, targets), 'cross_partitions': {}}
            for left, right in [('train', 'test'), ('fit', 'validation'), ('fit', 'test'), ('validation', 'test')]:
                projection['cross_partitions'][f'{left}_vs_{right}'] = overlap(
                    codes, partitions[left], partitions[right], targets)
            task_result['projections'][name] = projection
            cross = projection['cross_partitions']['train_vs_test']
            print(f'{task}/{name}: {cross["right_rows_with_left_match"]}/{cross["right_rows"]} '
                  'test rows have an equal training row', flush=True)
        result['tasks'][task] = task_result
    result['provenance']['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(args.output)

if __name__ == '__main__':
    main()
