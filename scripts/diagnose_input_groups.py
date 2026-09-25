from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import tensorflow as tf

from benchmark_int8_runtime import array_sha256, file_record, git_provenance
from data_loading import internal_fit_validation_indices, load_multiclass_split

def require(condition, message):
    if not condition:
        raise ValueError(message)

def quantize_dequantize(X, q):
    scale, zero = q['input_scale'], q['input_zero_point']
    require(np.isfinite(scale) and scale > 0, 'Invalid input quantization scale')
    rounded = np.round(X / scale + zero)
    clipped = np.clip(rounded, q['input_quantized_min'], q['input_quantized_max'])
    return ((clipped - zero) * scale).astype(np.float32), rounded != clipped

def input_cases(X, X_qdq, numeric, onehot):
    yield 'keras_fp32', X
    yield 'keras_qdq', X_qdq
    restored = X_qdq.copy()
    restored[:, numeric] = X[:, numeric]
    yield 'keras_restore_numeric', restored
    restored = X_qdq.copy()
    restored[:, onehot] = X[:, onehot]
    yield 'keras_restore_onehot', restored

def score(y, probabilities, class_map):
    require(probabilities.shape == (len(y), len(class_map)), 'Invalid prediction shape')
    require(np.isfinite(probabilities).all(), 'Non-finite predictions')
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((len(class_map), len(class_map)), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    per_class = []
    for i in range(len(class_map)):
        support = int(confusion[i].sum())
        per_class.append({
            'class_index': i, 'class_name': class_map[i], 'support': support,
            'correct': int(confusion[i, i]),
            'recall': float(confusion[i, i] / support) if support else None,
        })
    recalls = [item['recall'] for item in per_class if item['recall'] is not None]
    return {
        'accuracy': float(np.mean(predicted == y)),
        'correct': int(np.count_nonzero(predicted == y)),
        'macro_recall_present_classes': float(np.mean(recalls)),
        'per_class': per_class,
        'confusion_matrix_true_rows_predicted_columns': confusion.tolist(),
        'predictions_sha256': array_sha256(probabilities),
    }, predicted

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--csv', type=Path, default=Path('data/train_test_network.csv'))
    ap.add_argument('--reference-report', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    require(not args.output.exists(), f'ERROR: {args.output} already exists.')

    started = datetime.now(timezone.utc).isoformat()
    reference_path = args.reference_report or args.run_dir / 'int8_diagnostics/runtime_batch_comparison.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    require(reference['evaluation_split'] == 'internal_validation', 'Reference is not validation')
    directory = args.run_dir / 'multiclass/models/mlp_mc'
    int8_directory = args.run_dir / 'multiclass/models/mlp_mc_int8'
    paths = {
        'dataset': args.csv,
        'preprocessor': directory / 'preprocessor.pkl',
        'preprocessor_metadata': directory / 'preprocessor_meta.json',
        'scaler': directory / 'scaler.pkl',
        'keras_model': directory / 'model.keras',
        'int8_model': int8_directory / 'model.tflite',
        'calibration_statistics': int8_directory / 'calibration_statistics.json',
    }
    files = {name: file_record(path) for name, path in paths.items()}
    for name, record in files.items():
        require(record['sha256'] == reference['provenance']['files'][name]['sha256'],
                f'{name}: hash differs from reference report')
    require(files['int8_model']['sha256'] == reference['model_sha256'], 'INT8 model hash mismatch')
    meta = json.loads(paths['preprocessor_metadata'].read_text(encoding='utf-8'))
    train, _, labels, _, features, _, class_map = load_multiclass_split(
        str(args.csv), str(paths['preprocessor']), str(paths['preprocessor_metadata']), model_name='mlp_mc'
    )
    train, labels = np.asarray(train, dtype=np.float32), np.asarray(labels)
    _, validation = internal_fit_validation_indices(labels)
    n = reference['n_samples']
    require(0 < n <= len(validation), 'Invalid reference sample count')
    selected = np.asarray(validation[:n], dtype=np.int64)
    evaluation = reference['evaluation']
    require(selected.tolist() == evaluation['outer_training_positions'], 'Validation order mismatch')
    require(array_sha256(selected) == evaluation['indices_sha256'], 'Validation index hash mismatch')
    X = joblib.load(paths['scaler']).transform(train[selected]).astype(np.float32)
    y = labels[selected]
    require(array_sha256(X) == evaluation['inputs_sha256'], 'Input hash mismatch')
    require(array_sha256(y) == evaluation['labels_sha256'], 'Label hash mismatch')
    require(np.isfinite(X).all(), 'Non-finite input values')
    class_map = {int(k): str(v) for k, v in class_map.items()}
    require(class_map == {int(k): v for k, v in evaluation['class_map'].items()}, 'Class map mismatch')
    require(sorted(class_map) == list(range(len(class_map))), 'Non-contiguous class indices')
    require(np.issubdtype(y.dtype, np.integer) and np.isin(y, list(class_map)).all(), 'Invalid labels')
    require(list(features) == meta['feature_names'], 'Feature order mismatch')
    contract = meta['post_preprocessing_scaler']
    require(contract['kind'] == 'numeric_only_standard_scaler', 'Unexpected scaling contract')
    require(contract['scaled_features'] == meta['numeric_features'], 'Numeric scaling contract mismatch')
    numeric = [i for i, name in enumerate(features) if name in set(meta['numeric_features'])]
    onehot = [i for i in range(len(features)) if i not in set(numeric)]
    require(numeric and onehot, 'Both numeric and one-hot groups must be present')
    require(np.isin(X[:, onehot], [0.0, 1.0]).all(), 'One-hot inputs are not 0/1')
    q = reference['runs'][0]['runtimes']['tflite']['input_quantization']
    quant_keys = ['input_scale', 'input_zero_point', 'input_quantized_min', 'input_quantized_max']
    for batch in reference['runs']:
        for runtime in batch['runtimes'].values():
            require(all(runtime['input_quantization'][k] == q[k] for k in quant_keys),
                    'Reference quantization differs across runtime/batch configurations')
    interpreter = tf.lite.Interpreter(model_path=str(paths['int8_model']))
    interpreter.allocate_tensors()
    detail = interpreter.get_input_details()[0]
    require(np.issubdtype(detail['dtype'], np.integer), 'Model input is not integer')
    limits = np.iinfo(detail['dtype'])
    require(tuple(detail['quantization']) == (q['input_scale'], q['input_zero_point'])
            and limits.min == q['input_quantized_min'] and limits.max == q['input_quantized_max'],
            'Model quantization differs from reference')
    X_qdq, clipping = quantize_dequantize(X, q)
    require(int(clipping.sum()) == q['clipped_count'], 'Clipping count differs from reference')
    model = tf.keras.models.load_model(paths['keras_model'])
    cases, predicted = {}, {}
    for name, current in input_cases(X, X_qdq, numeric, onehot):
        cases[name], predicted[name] = score(y, np.asarray(model.predict(current, verbose=0)), class_map)
        print(f'{name}: accuracy={cases[name]["accuracy"]:.4%}', flush=True)
        if name in ('keras_fp32', 'keras_qdq'):
            expected = (reference['keras_fp32_accuracy'] if name == 'keras_fp32'
                        else reference['runs'][0]['runtimes']['tflite']['keras_qdq_accuracy'])
            require(np.isclose(cases[name]['accuracy'], expected, rtol=0, atol=1e-12),
                    f'{name} - baseline differs from reference, do not interpret restoration results')
    for name, metrics in cases.items():
        metrics['accuracy_delta_pp_vs_qdq'] = 100 * (metrics['accuracy'] - cases['keras_qdq']['accuracy'])
        metrics['changed_predictions_vs_qdq'] = int(np.count_nonzero(predicted[name] != predicted['keras_qdq']))
        metrics['corrected_vs_qdq'] = int(np.count_nonzero((predicted[name] == y) & (predicted['keras_qdq'] != y)))
        metrics['regressed_vs_qdq'] = int(np.count_nonzero((predicted[name] != y) & (predicted['keras_qdq'] == y)))
    feature_errors = []
    for i, name in enumerate(features):
        difference = np.abs(X[:, i] - X_qdq[:, i])
        feature_errors.append({
            'index': i, 'feature': name, 'group': 'numeric' if i in numeric else 'onehot',
            'validation_min': float(X[:, i].min()), 'validation_max': float(X[:, i].max()),
            'mean_abs_error': float(difference.mean()), 'max_abs_error': float(difference.max()),
            'clipped_values': int(clipping[:, i].sum()),
            'nonzero_values_rounded_to_zero': int(np.count_nonzero((X[:, i] != 0) & (X_qdq[:, i] == 0))),
        })
    result = {
        'diagnostic_type': 'keras_input_group_restoration',
        'interpretation': 'Hybrid Keras inputs only',
        'evaluation_split': 'internal_validation', 'n_samples': n,
        'reference_report': file_record(reference_path), 'evaluation': evaluation,
        'numeric_features': [features[i] for i in numeric],
        'onehot_features': [features[i] for i in onehot],
        'input_quantization': {k: q[k] for k in quant_keys},
        'cases': cases, 'feature_quantization_errors': feature_errors,
        'provenance': {
            'started_at_utc': started, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'command_argv': [sys.executable, *sys.argv], 'cwd': str(Path.cwd()),
            'git': git_provenance(Path(__file__).resolve().parent.parent, args.run_dir),
            'platform': platform.platform(), 'python': platform.python_version(),
            'tensorflow': tf.__version__, 'numpy': np.__version__, 'files': files,
            'script': file_record(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(args.output)

if __name__ == '__main__':
    main()
