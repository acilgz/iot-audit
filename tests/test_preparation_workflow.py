import importlib
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from data_loading import (TARGET_ALIASES, load_binary_split, load_multiclass_split,
                          load_preprocessor, model_paths, prepare, validate_model, validate_metrics)

ROOT = Path(__file__).resolve().parents[1]

def run_script(name, *args):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / name), *map(str, args)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    return result

@pytest.fixture
def csv(tmp_path):
    n = 120
    frame = pd.DataFrame({'duration': np.arange(n, dtype=float),
                          'bytes': np.arange(n, dtype=float) ** 2,
                          'proto': ['tcp', 'udp', 'icmp'] * 40,
                          'label': [0, 1] * 60,
                          'type': ['normal', 'dos', 'scan'] * 40})
    for col in TARGET_ALIASES - {'label', 'type'}:
        frame[col] = np.arange(n) % 3
    path = tmp_path / 'data.csv'
    frame.to_csv(path, index=False)
    return path

@pytest.mark.parametrize('mode,suffix', [('binary', ''), ('multiclass', '_mc')])
def test_cli_training_roundtrip_and_benchmark(tmp_path, csv, mode, suffix):
    out = tmp_path / mode
    names = ['logreg' + suffix, 'rf' + suffix]
    run_script('prepare_preprocessor' + suffix + '.py', '--csv', csv, '--outdir', out,
               '--models', *names)
    loader = load_binary_split if mode == 'binary' else load_multiclass_split
    benchmark = importlib.import_module('compare_models' + suffix)
    df = pd.read_csv(csv)
    for family, name in zip(['logreg', 'rf'], names):
        script = ('train_' + family if not suffix else 'train_mc_' + family) + '.py'
        extra = ['--n_estimators', '3', '--max_depth', '3'] if family == 'rf' else []
        run_script(script, '--csv', csv, '--outdir', out, *extra)
        pp, mp = model_paths(out, name)
        preproc, meta = load_preprocessor(pp, mp, name, mode)
        assert isinstance(preproc.named_transformers_['num'].named_steps['scaler'], StandardScaler) == (family == 'logreg')
        assert not TARGET_ALIASES.intersection(meta['numeric_features'] + meta['categorical_features'])
        validate_model(Path(pp).parent)
        result = loader(csv, pp, mp, model_name=name)
        model = joblib.load(Path(pp).parent / 'model.pkl')
        expected = model.predict(pd.DataFrame(preproc.transform(df), columns=meta['feature_names']))
        modified = df.copy()
        for col in TARGET_ALIASES:
            modified[col] = 'changed'
        for variant in [df.drop(columns=list(TARGET_ALIASES)), modified]:
            transformed = pd.DataFrame(preproc.transform(variant), columns=meta['feature_names'])
            np.testing.assert_allclose(transformed, preproc.transform(df))
            np.testing.assert_array_equal(model.predict(transformed), expected)
        metrics = json.loads((Path(pp).parent / 'metrics.json').read_text())
        cm = np.asarray(metrics['confusion_matrix'])
        assert cm.ndim == 2 and cm.shape[0] == cm.shape[1]
        assert cm.sum() == len(result[3])
        assert metrics['accuracy'] == pytest.approx(cm.trace() / cm.sum())
        with pytest.raises(FileExistsError):
            loader(csv, pp, mp, model_name=name, for_training=True)
    for index, frame in enumerate([df, df.drop(columns=list(TARGET_ALIASES)), modified]):
        path = tmp_path / f'benchmark-{index}.csv'
        frame.to_csv(path, index=False)
        measured = benchmark.benchmark_inference(str(out), str(path), names, sample_size=12, num_runs=2)
        assert len(measured) == 6
        assert set(measured['model']) == set(names)
        assert set(measured['n_samples']) == {12}
    benchout = tmp_path / ('bench-cli' + suffix)
    run_script('compare_models' + suffix + '.py', '--models-dir', out, '--outdir', benchout,
               '--csv', csv, '--models', *names, '--benchmark', '--device-id', 'test-device',
               '--sample_size', '12', '--num_runs', '2')
    sidecar = json.loads((benchout / ('inference_benchmark' + suffix + '.manifest.json')).read_text())
    assert sidecar['device_id'] == 'test-device'
    assert sidecar['num_runs'] == 2
    assert set(sidecar['models']) == set(names)
    bundle = out / 'models' / names[0]
    with open(bundle / 'model.pkl', 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_model(bundle)
    with pytest.raises(FileExistsError):
        prepare(csv, out, names, mode)

@pytest.mark.parametrize('mode,suffix', [('binary', ''), ('multiclass', '_mc')])
def test_contract_rejects_dataset_split_and_bundle_changes(tmp_path, csv, mode, suffix):
    out = tmp_path / mode
    name = 'logreg' + suffix
    prepare(csv, out, [name], mode)
    pp, mp = model_paths(out, name)
    loader = load_binary_split if mode == 'binary' else load_multiclass_split
    with pytest.raises(ValueError, match='Split configuration'):
        loader(csv, pp, mp, random_state=1)
    with pytest.raises(ValueError, match='model/task'):
        load_preprocessor(pp, mp, 'rf' + suffix, mode)
    altered = tmp_path / 'altered.csv'
    df = pd.read_csv(csv)
    df.loc[0, 'duration'] = 9999
    df.to_csv(altered, index=False)
    with pytest.raises(ValueError, match='Dataset differs'):
        loader(altered, pp, mp)
    with open(pp, 'ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_preprocessor(pp, mp, name, mode)

@pytest.mark.parametrize('mode,suffix', [('binary', ''), ('multiclass', '_mc')])
def test_prepare_all_models_keeps_mlp_scaler_separate(tmp_path, csv, mode, suffix):
    names = [name + suffix for name in ['logreg', 'rf', 'xgb', 'lgbm', 'mlp']]
    prepare(csv, tmp_path / mode, names, mode)
    contracts = []
    for name in names:
        pp, mp = model_paths(tmp_path / mode, name)
        _, meta = load_preprocessor(pp, mp, name, mode)
        assert meta['scale_numeric'] == name.startswith('logreg')
        contracts.append(meta)
    for meta in contracts[1:]:
        assert meta['feature_names'] == contracts[0]['feature_names']
        assert meta['train_indices'] == contracts[0]['train_indices']
        assert meta['test_indices'] == contracts[0]['test_indices']

@pytest.mark.parametrize('metrics', [
    {'accuracy': 1, 'confusion_matrix': [[3, 0, 1], [0, 3]]},
    {'accuracy': .99, 'confusion_matrix': [[3, 1], [1, 3]]},
    {'accuracy': 1, 'confusion_matrix': [[-1, 0], [0, 3]]},
])
def test_rejects_merge_damaged_metrics(metrics):
    with pytest.raises(ValueError):
        validate_metrics(metrics)

def test_model_formats_are_ignored():
    names = ['x.pkl', 'x.joblib', 'x.keras', 'x.h5', 'x.tflite', 'x.onnx', 'x.pb',
             'benchmark/test/model.pkl', 'benchmark/test/model.keras', 'benchmark/test/model.tflite']
    result = subprocess.run(['git', 'check-ignore', '--stdin'], input='\n'.join(names),
                            capture_output=True, text=True, cwd=ROOT)
    if result.returncode == 128:
        pytest.skip('Git metadata unavailable')
    assert set(result.stdout.splitlines()) == set(names)
