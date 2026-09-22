
from __future__ import annotations
import os, sys, json, time, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from iot_audit.metrics import evaluate_model
from data_loading import load_binary_split
import numpy as np
import joblib
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class KerasSklearnWrapper:
    def __init__(self, keras_model):
        self.keras_model = keras_model
        self.feature_importances_ = None

    def predict_proba(self, X):
        p1 = self.keras_model.predict(X, verbose=0).reshape(-1)
        return np.stack([1 - p1, p1], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class TFLiteInt8Wrapper:
    def __init__(self, tflite_model):
        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()
        self.in_d = self.interpreter.get_input_details()[0]
        self.out_d = self.interpreter.get_output_details()[0]
        self.feature_importances_ = None

    def predict_proba(self, X):
        in_scale, in_zero = self.in_d["quantization"]
        out_scale, out_zero = self.out_d["quantization"]
        qmin, qmax = np.iinfo(self.in_d["dtype"]).min, np.iinfo(self.in_d["dtype"]).max

        X = np.asarray(X, dtype=np.float32)
        X_q = np.clip(np.round(X / in_scale + in_zero), qmin, qmax).astype(self.in_d["dtype"])

        p1 = np.empty(len(X), dtype=np.float32)
        for i in range(len(X)):
            self.interpreter.set_tensor(self.in_d["index"], X_q[i:i + 1])
            self.interpreter.invoke()
            out_q = self.interpreter.get_tensor(self.out_d["index"])
            p1[i] = (out_q.astype(np.float32) - out_zero) * out_scale
        return np.stack([1 - p1, p1], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--outdir", default="train")
    ap.add_argument("--n_count", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--calib_samples", type=int, default=1000)
    args = ap.parse_args()
    preproc_path = os.path.join(args.outdir, "preprocessor", "preprocessor.pkl")
    meta_path = os.path.join(os.path.dirname(preproc_path), "preprocessor_meta.json")

    model_name = "mlp"

    X_train, X_test, y_train, y_test, feature_names, preproc = load_binary_split(
        args.csv, preproc_path, meta_path
    )
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"]
    )

    print(f"[mlp_int8] training ({X_train.shape[0]} samples, {X_train.shape[1]} features)...")
    t0 = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.n_count,
        batch_size=args.batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)],
        verbose=2
    )
    print(f"[mlp_int8] done in {time.time()-t0:.2f}s")

    model_dir = os.path.join(args.outdir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.keras"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    wrapped = KerasSklearnWrapper(model)
    y_pred = wrapped.predict(X_test)
    y_proba = wrapped.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(
        y_test, y_pred, y_proba, feature_names, wrapped,
        model_name=model_name, base_outdir=args.outdir
    )
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{model_name}] metrics (float):", json.dumps(metrics, indent=2))



if __name__ == "__main__":
    main()