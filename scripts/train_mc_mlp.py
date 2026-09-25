from __future__ import annotations
import os, sys, json, time, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from iot_audit.metrics_mc import evaluate_model_multiclass
from data_loading import model_paths, record_model, load_multiclass_split
import numpy as np
import joblib
import argparse
import tensorflow as tf
from data_loading import internal_fit_validation_indices, NumericOnlyScaler

class KerasSoftmaxWrapper:
    def __init__(self, keras_model):
        self.keras_model = keras_model
        self.feature_importances_ = None

    def predict_proba(self, X):
        return self.keras_model.predict(X, verbose=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class TFLiteInt8MulticlassWrapper:
    def __init__(self, tflite_model):
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()
        self.in_d = self.interpreter.get_input_details()[0]
        self.out_d = self.interpreter.get_output_details()[0]
        self.n_classes = self.out_d["shape"][-1]
        self.feature_importances_ = None

    def predict_proba(self, X):
        in_scale, in_zero = self.in_d["quantization"]
        out_scale, out_zero = self.out_d["quantization"]
        qmin, qmax = np.iinfo(self.in_d["dtype"]).min, np.iinfo(self.in_d["dtype"]).max

        X = np.asarray(X, dtype=np.float32)
        X_q = np.clip(np.round(X / in_scale + in_zero), qmin, qmax).astype(self.in_d["dtype"])

        probs = np.empty((len(X), self.n_classes), dtype=np.float32)
        for i in range(len(X)):
            self.interpreter.set_tensor(self.in_d["index"], X_q[i:i + 1])
            self.interpreter.invoke()
            out_q = self.interpreter.get_tensor(self.out_d["index"])
            probs[i] = (out_q.astype(np.float32) - out_zero) * out_scale
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--outdir", default="train_mc")
    ap.add_argument("--n_count", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--calib_samples", type=int, default=1000)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(42)
    model_name = "mlp_mc"
    preproc_path, meta_path = model_paths(args.outdir, model_name)

    X_train, X_test, y_train, y_test, feature_names, preproc, class_map = load_multiclass_split(
        args.csv, preproc_path, meta_path, model_name=model_name, for_training=True
    )
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    num_classes = len(class_map)

    fit_idx, val_idx = internal_fit_validation_indices(y_train)
    scaler = NumericOnlyScaler(feature_names, preproc.numeric_features_)
    scaler.fit(X_train[fit_idx])
    X_fit = scaler.transform(X_train[fit_idx])
    X_val = scaler.transform(X_train[val_idx])
    X_test = scaler.transform(X_test)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"[mlp_int8_mc] training ({len(fit_idx)} fit + {len(val_idx)} validation samples, {X_train.shape[1]} features, classes={num_classes})...")
    t0 = time.time()
    model.fit(
        X_fit, y_train[fit_idx],
        validation_data=(X_val, y_train[val_idx]),
        epochs=args.n_count,
        batch_size=args.batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True)],
        verbose=2
    )
    print(f"[mlp_int8_mc] done in {time.time()-t0:.2f}s")

    model_dir = os.path.join(args.outdir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.keras"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    wrapped = KerasSoftmaxWrapper(model)
    y_pred = wrapped.predict(X_test)
    y_proba = wrapped.predict_proba(X_test)
    metrics = evaluate_model_multiclass(
        y_test, y_pred, y_proba, feature_names, wrapped,
        class_map=class_map, model_name=model_name, base_outdir=args.outdir
    )
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{model_name}] metrics (float):", json.dumps(metrics, indent=2))

    record_model(model_dir)

if __name__ == "__main__":
    main()