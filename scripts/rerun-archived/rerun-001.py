# Re-runs 001 with the original 80/20 split and outer-training calibration
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

TRAIN_COMMIT = 'b286c11df3b8aa39f6094c803dc4ab845d44cdf5'
QUANT_SOURCE_SHA = 'ecbe65102f263b2897c4117eea88aad83b3d43cb384dc9b6c29217588ede3783'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def error_stats(a, b):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return {'max_abs_error': float(np.abs(d).max()), 'mean_abs_error': float(np.abs(d).mean()),
            'rmse': float(np.sqrt(np.mean(d*d)))}

def git(repo, *arguments):
    return subprocess.run(['git', '-C', str(repo), *arguments], check=True, capture_output=True).stdout

def predict_tflite(tf, path, X):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    input_scale, input_zero = inp['quantization']
    output_scale, output_zero = out['quantization']
    integer = np.issubdtype(inp['dtype'], np.integer)
    if integer:
        limits = np.iinfo(inp['dtype'])
        require(input_scale > 0, 'Invalid input scale')
        inputs = np.clip(np.round(X / input_scale + input_zero), limits.min, limits.max).astype(inp['dtype'])
    else:
        inputs = X.astype(inp['dtype'])
    predictions = np.empty((len(X), int(out['shape'][-1])), dtype=np.float32)
    for i in range(len(X)):
        interpreter.set_tensor(inp['index'], inputs[i:i+1])
        interpreter.invoke()
        p = interpreter.get_tensor(out['index'])
        if np.issubdtype(out['dtype'], np.integer):
            p = (p.astype(np.float32) - output_zero) * output_scale
        predictions[i] = p[0]
    return predictions, {'input': {'dtype': str(inp['dtype']), 'scale': float(input_scale), 'zero_point': int(input_zero)},
                         'output': {'dtype': str(out['dtype']), 'scale': float(output_scale), 'zero_point': int(output_zero)}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/001'))
    parser.add_argument('--csv', type=Path, default=Path('data/train_test_network.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/run001-recheck'))
    args = parser.parse_args()
    output = args.output_dir / 'run001-recheck.json'
    require(not output.exists(), f'Report already exists: {output}')
    require(not args.output_dir.resolve().is_relative_to(args.run_dir.resolve()), 'Output must be outside runs/001')
    started = datetime.now(timezone.utc).isoformat()
    base = args.run_dir / 'multiclass/models'
    kd, qd = base / 'mlp_mc', base / 'mlp_mc_int8'
    hd = args.run_dir / 'int8_diagnostics'
    meta = json.loads((kd/'preprocessor_meta.json').read_text())
    require(meta['commit'] == TRAIN_COMMIT, 'does not match the original run001 preprocessing contract')
    require(sha(args.csv) == meta['dataset_sha256'], 'Dataset hash mismatch')
    require(sklearn.__version__ == meta['environment']['scikit-learn'],
            'Original scikit-learn environment: ' + meta['environment']['scikit-learn'])
    require(np.__version__ == meta['environment']['numpy'], 'Original NumPy environment')
    import tensorflow as tf
    require(tf.__version__ == '2.16.2', 'TensorFlow 2.16.2 required')
    source_files = {'dataset': record(args.csv)}
    for name, folder in [('keras', kd), ('int8', qd)]:
        manifest = json.loads((folder/'model_manifest.json').read_text())
        for filename, expected in manifest['files'].items():
            require(sha(folder/filename) == expected, f'Manifest mismatch: {folder/filename}')
            source_files[f'{name}/{filename}'] = record(folder/filename)
        if name == 'int8':
            conversion_command = manifest['command']
        source_files[f'{name}/model_manifest.json'] = record(folder/'model_manifest.json')
    for filename in ['metrics.json', 'quantization.json', 'input_statistics.json', 'model_fp32.tflite']:
        source_files[f'historical_diagnostic/{filename}'] = record(hd/filename)
    repo = args.run_dir.resolve().parent.parent
    quant_source = git(repo, 'show', TRAIN_COMMIT+':scripts/quantize_model_mc.py')
    require(hashlib.sha256(quant_source).hexdigest() == QUANT_SOURCE_SHA, 'Unexpected historical calibration source')
    count = int(conversion_command[conversion_command.index('--calib_samples')+1])
    require(count == 1000, 'Unexpected historical calibration sample count')
    try:
        df = pd.read_csv(args.csv, engine='pyarrow')
    except ImportError:
        df = pd.read_csv(args.csv)
    classes, y = np.unique(df['type'].astype(str).str.strip(), return_inverse=True)
    require({str(i):str(c) for i,c in enumerate(classes)} == meta['class_map'], 'Class map mismatch')
    train = np.asarray(meta['train_indices'], dtype=np.int64)
    test = np.asarray(meta['test_indices'], dtype=np.int64)
    require(np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(df))),
            'Saved train/test indices must be disjoint partitions of the dataset')
    preprocessor = joblib.load(kd/'preprocessor.pkl')
    scaler = joblib.load(kd/'scaler.pkl')
    predictors = df.drop(columns=[c for c in df.columns if c in {'label','Label','LABEL','type','Type','TYPE','target','Target','TARGET'}])
    names = meta['feature_names']
    def transform(rows):
        raw = preprocessor.transform(predictors.iloc[rows])
        # Scale float64 preprocessed DataFrame before casting, like the original code
        return scaler.transform(pd.DataFrame(raw, columns=names)).astype(np.float32)
    X_train, X_test = transform(train), transform(test)
    historical = json.loads((hd/'input_statistics.json').read_text())
    input_stats = {'min':float(X_test.min()), 'max':float(X_test.max()),
                   'percentile_99':float(np.percentile(X_test,99)),
                   'abs_gt_100':int((np.abs(X_test)>100).sum()), 'abs_gt_200':int((np.abs(X_test)>200).sum())}
    require(all(input_stats[k] == v for k,v in historical.items()), 'Archived test input statistics do not match')
    positions = np.random.default_rng(42).choice(len(train), size=count, replace=False)
    calibration = X_train[positions]
    print('Calibration reconstructed:',float(calibration.min()),float(calibration.max()),flush=True)
    model = tf.keras.models.load_model(kd/'model.keras')
    probabilities = {'keras_fp32':model.predict(X_test,verbose=0)}
    probabilities['tflite_fp32'], _ = predict_tflite(tf,hd/'model_fp32.tflite',X_test)
    probabilities['tflite_int8'], quant = predict_tflite(tf,qd/'model.tflite',X_test)
    require(quant == json.loads((hd/'quantization.json').read_text()), 'Archived model quantization parameters do not match')
    scale, zero = quant['input']['scale'], quant['input']['zero_point']
    unbounded = np.round(X_test/scale+zero)
    bounded = np.clip(unbounded,-128,127)
    X_qdq = (bounded-zero)*scale
    probabilities['keras_qdq'] = model.predict(X_qdq,verbose=0)
    archived = {item['model']:item for item in json.loads((hd/'metrics.json').read_text())}
    metrics = {}
    for name, probs in probabilities.items():
        require(probs.shape == (len(test),len(classes)) and np.isfinite(probs).all(), 'Invalid predictions')
        pred = np.argmax(probs,axis=1)
        report = classification_report(y[test],pred,labels=np.arange(len(classes)),output_dict=True,zero_division=0)
        metrics[name] = {'accuracy':float(accuracy_score(y[test],pred)), 'macro_f1':float(f1_score(y[test],pred,average='macro')),
            'per_class':[{'index':i,'name':str(c),'recall':float(report[str(i)]['recall']),'support':int(report[str(i)]['support'])} for i,c in enumerate(classes)],
            'confusion_matrix':confusion_matrix(y[test],pred,labels=np.arange(len(classes))).tolist(),
            'disagreements_with_archived_predictions':int(np.count_nonzero(pred != np.asarray(archived[name]['predictions'])))}
        print(name,metrics[name]['accuracy'],'archived disagreements',metrics[name]['disagreements_with_archived_predictions'],flush=True)
    comparisons = {}
    for left,right in [('keras_fp32','tflite_fp32'),('keras_fp32','keras_qdq'),('keras_fp32','tflite_int8'),('keras_qdq','tflite_int8')]:
        a,b = probabilities[left], probabilities[right]
        comparisons[left+'__'+right] = {**error_stats(a,b), 'class_disagreements':int(np.count_nonzero(a.argmax(1)!=b.argmax(1)))}
    clipping = unbounded != bounded
    input_stats.update({'n_samples':len(test),'n_features':len(names),'input_count':int(X_test.size),
        'clipped_values':int(clipping.sum()),'clipped_rows':int(np.any(clipping,axis=1).sum()),
        'clipped_fraction':float(clipping.mean()),'saturation_min':int((bounded==-128).sum()),'saturation_max':int((bounded==127).sum()),
        'quantization_error':error_stats(X_test,X_qdq),
        'real_range':[( -128-zero)*scale,(127-zero)*scale],
        'test_dataset_indices':test.tolist()})
    for item in source_files.values():
        require(sha(item['path']) == item['sha256'], 'Historical source changed during recheck')
    result = {'analysis_scope':'retrospective_test_run001_no_tuning','source_files':source_files,
        'historical_provenance':{'training_commit':TRAIN_COMMIT,'training_git_status':(args.run_dir/'git-status.txt').read_text(),
            'conversion_command_as_recorded':conversion_command,
            'conversion_source_sha256':QUANT_SOURCE_SHA,
            'original_diagnostic_command_and_platform':'Not captured in the archived diagnostic'},
        'recheck_provenance':{'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
            'command_argv':[sys.executable,*sys.argv],'cwd':str(Path.cwd()),'platform':platform.uname()._asdict(),
            'git_commit':git(repo,'rev-parse','HEAD').decode().strip(),'git_status':git(repo,'status','--porcelain').decode(),
            'versions':{'python':platform.python_version(),'tensorflow':tf.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scikit-learn':sklearn.__version__,
                        'keras':importlib.metadata.version('keras')},'script':record(Path(__file__))},
        'calibration':{'status':'reconstructed from original artifacts and conversion source',
            'source':'original outer training partition','seed':42,'used_samples':count,'training_rows':len(train),
            'min':float(calibration.min()),'max':float(calibration.max()),
            'expected_input_scale_float32':float(np.float32((float(calibration.max())-float(calibration.min()))/255)),
            'input_scale_matches_reconstructed_range':float(np.float32((float(calibration.max())-float(calibration.min()))/255))==scale,
            'sample_dataset_indices':train[positions].tolist(),
            'per_feature':[{'index':i,'name':n,'min':float(calibration[:,i].min()),'max':float(calibration[:,i].max())} for i,n in enumerate(names)]},
        'quantization':quant,'input_statistics':input_stats,'metrics':metrics,'output_comparisons':comparisons,
        'archived_predictions_reproduced':all(v['disagreements_with_archived_predictions']==0 for v in metrics.values()),
        'historical_sources_unchanged':True}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(output)
    if not result['archived_predictions_reproduced']:
        raise SystemExit('ERROR: Failed to reproduce archived predictions.')

if __name__ == '__main__':
    main()
