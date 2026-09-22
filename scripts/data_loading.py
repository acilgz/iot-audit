from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from iot_audit.preprocessing import SCALE, _read_csv, _normalize_label, load_and_prepare_data
from iot_audit.preprocessing_mc import load_and_prepare_multiclass

TARGET_ALIASES = {'label', 'Label', 'LABEL', 'type', 'Type', 'TYPE', 'target', 'Target', 'TARGET'}
MODEL_FILES = ('model.pkl', 'model.keras', 'model.tflite')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def model_paths(outdir, model_name):
    directory = Path(outdir) / 'models' / model_name
    return str(directory / 'preprocessor.pkl'), str(directory / 'preprocessor_meta.json')


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def prepare(csv_path, outdir, model_names, mode):
    for name in model_names:
        destination = Path(outdir) / 'models' / name
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f'{destination} is not empty')
    dataset_hash = sha256(csv_path)
    df = _read_csv(csv_path)
    y = _normalize_label(df['label']).to_numpy() if mode == 'binary' else np.unique(df['type'].astype(str).str.strip(), return_inverse=True)[1]
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=.2, random_state=42, stratify=y)
    repo = Path(__file__).resolve().parents[1]
    def git(*args):
        p = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    for name in model_names:
        if mode == 'binary':
            *_, features, preproc = load_and_prepare_data(csv_path, leakage_base=outdir, model_name=name)
            class_map = {}
        else:
            *_, features, preproc, class_map = load_and_prepare_multiclass(csv_path, base_outdir=outdir, model_name=name)
        pp, mp = model_paths(outdir, name)
        Path(pp).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preproc, pp)
        _write_json(mp, {
            'schema_version': 1, 'mode': mode, 'model_name': name,
            'dataset_sha256': dataset_hash, 'preprocessor_sha256': sha256(pp),
            'feature_names': features, 'numeric_features': preproc.numeric_features_,
            'categorical_features': preproc.categorical_features_, 'class_map': class_map,
            'scale_numeric': SCALE[name], 'test_size': .2, 'random_state': 42,
            'train_indices': train_idx.tolist(), 'test_indices': test_idx.tolist(),
            'commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
            'command': sys.argv, 'environment': {'python': platform.python_version(),
            'numpy': np.__version__, 'pandas': pd.__version__, 'scikit-learn': sklearn.__version__},
        })
        print(f'[prepare] {name}: {pp}')


def load_preprocessor(preproc_path, meta_path, model_name=None, mode=None):
    if not Path(preproc_path).is_file() or not Path(meta_path).is_file():
        raise FileNotFoundError('Missing per-model preprocessor/contract. Run prepare_preprocessor in a new directory')
    meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
    if meta.get('schema_version') != 1:
        raise ValueError('Unsupported or legacy preprocessing contract')
    source_name = meta['model_name']
    expected_name = model_name.removesuffix('_int8') if model_name else source_name
    if source_name != expected_name or (mode and meta['mode'] != mode):
        raise ValueError('Preprocessor model/task mismatch')
    if sha256(preproc_path) != meta['preprocessor_sha256']:
        raise ValueError('Preprocessor hash mismatch')
    if meta['environment']['scikit-learn'] != sklearn.__version__:
        raise ValueError('scikit-learn version differs from preprocessing environment')
    preproc = joblib.load(preproc_path)
    num, cat = preproc.numeric_features_, preproc.categorical_features_
    fitted = {n: list(cols) for n, _, cols in preproc.transformers_ if n != 'remainder'}
    if fitted != {'num': num, 'cat': cat} or preproc.remainder != 'drop':
        raise ValueError('Fitted transformer columns differ from contract')
    if TARGET_ALIASES.intersection(num + cat):
        raise ValueError('Target leakage in preprocessor')
    if num != meta['numeric_features'] or cat != meta['categorical_features']:
        raise ValueError('Feature contract mismatch')
    scaled = isinstance(preproc.named_transformers_['num'].named_steps['scaler'], StandardScaler)
    if scaled != SCALE[source_name] or scaled != meta['scale_numeric']:
        raise ValueError('Scaling policy mismatch')
    ohe = preproc.named_transformers_['cat'].named_steps['onehot'] if cat else None
    features = num + (ohe.get_feature_names_out(cat).tolist() if cat else [])
    if features != meta['feature_names']:
        raise ValueError('Transformed feature order mismatch')
    return preproc, meta


def _load_split(csv_path, preproc_path, meta_path, mode, test_size, random_state, model_name, for_training):
    directory = Path(preproc_path).parent
    if for_training and any((directory / name).exists() for name in MODEL_FILES):
        raise FileExistsError(f'{directory} already contains a model; use a new run')
    preproc, meta = load_preprocessor(preproc_path, meta_path, model_name, mode)
    if sha256(csv_path) != meta['dataset_sha256']:
        raise ValueError('Dataset differs from preparation dataset')
    if test_size != meta['test_size'] or random_state != meta['random_state']:
        raise ValueError('Split configuration differs from preprocessing contract')
    df = _read_csv(csv_path)
    if mode == 'binary':
        y = _normalize_label(df['label']).to_numpy()
        class_map = {}
    else:
        classes, y = np.unique(df['type'].astype(str).str.strip(), return_inverse=True)
        class_map = {i: str(c) for i, c in enumerate(classes)}
        if class_map != {int(k): v for k, v in meta['class_map'].items()}:
            raise ValueError('Class map mismatch')
    train, test = train_test_split(np.arange(len(df)), test_size=test_size, random_state=random_state, stratify=y)
    if train.tolist() != meta['train_indices'] or test.tolist() != meta['test_indices']:
        raise ValueError('Split indices mismatch')
    X = df.drop(columns=list(TARGET_ALIASES), errors='ignore')
    features = meta['feature_names']
    result = (pd.DataFrame(preproc.transform(X.iloc[train]), columns=features),
              pd.DataFrame(preproc.transform(X.iloc[test]), columns=features),
              y[train], y[test], features, preproc)
    return result if mode == 'binary' else (*result, class_map)


def load_binary_split(csv_path, preproc_path, meta_path, test_size=.2, random_state=42, *, model_name=None, for_training=False):
    return _load_split(csv_path, preproc_path, meta_path, 'binary', test_size, random_state, model_name, for_training)


def load_multiclass_split(csv_path, preproc_path, meta_path, test_size=.2, random_state=42, *, model_name=None, for_training=False):
    return _load_split(csv_path, preproc_path, meta_path, 'multiclass', test_size, random_state, model_name, for_training)


def record_model(directory):
    directory = Path(directory)
    names = [n for n in (*MODEL_FILES, 'preprocessor.pkl', 'preprocessor_meta.json', 'scaler.pkl', 'metrics.json') if (directory / n).is_file()]
    _write_json(directory / 'model_manifest.json', {'files': {n: sha256(directory / n) for n in names}, 'command': sys.argv})


def validate_model(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'model_manifest.json').read_text(encoding='utf-8'))
    expected = manifest['files']
    actual = {n for n in (*MODEL_FILES, 'preprocessor.pkl', 'preprocessor_meta.json', 'scaler.pkl', 'metrics.json') if (directory / n).is_file()}
    required = {'preprocessor.pkl', 'preprocessor_meta.json', 'metrics.json'}
    if set(expected) != actual or not required.issubset(expected) or sum(n in expected for n in MODEL_FILES) != 1:
        raise ValueError('Model bundle is incomplete or contains unexpected artifacts')
    for name, digest in expected.items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Model bundle hash mismatch: {name}')
    meta = json.loads((directory / 'preprocessor_meta.json').read_text(encoding='utf-8'))
    if meta['model_name'].startswith('mlp') and 'scaler.pkl' not in expected:
        raise ValueError('MLP bundle requires its fitted scaler')
    validate_metrics(json.loads((directory / 'metrics.json').read_text(encoding='utf-8')))


def validate_metrics(metrics):
    cm = np.asarray(metrics['confusion_matrix'], dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.size == 0:
        raise ValueError('Confusion matrix must be square')
    if not np.isfinite(cm).all() or (cm < 0).any() or not np.equal(cm, np.floor(cm)).all() or cm.sum() == 0:
        raise ValueError('Invalid confusion matrix counts')
    if not np.isclose(cm.trace() / cm.sum(), metrics['accuracy'], rtol=0, atol=1e-12):
        raise ValueError('Accuracy does not match confusion matrix')


def record_benchmark(args, result_path):
    from datetime import datetime, timezone
    from importlib.metadata import version, PackageNotFoundError
    versions = {}
    for package in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'lightgbm', 'xgboost', 'tensorflow', 'keras', 'ai-edge-litert'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    manifests = {}
    for name in args.models:
        path = Path(args.models_dir) / 'models' / name / 'model_manifest.json'
        manifests[name] = json.loads(path.read_text(encoding='utf-8'))
    _write_json(Path(result_path).with_suffix('.manifest.json'), {
        'device_id': args.device_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'machine': platform.machine(), 'processor': platform.processor(),
        'python': platform.python_version(), 'packages': versions,
        'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS')},
        'dataset_sha256': sha256(args.csv), 'result_sha256': sha256(result_path),
        'sample_size': args.sample_size, 'sampling_seed': 42, 'num_runs': args.num_runs,
        'observation_unit': 'physical_device', 'command': sys.argv, 'models': manifests,
    })
