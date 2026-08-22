
from __future__ import annotations
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from iot_audit.metrics import evaluate_model
from data_loading import load_binary_split
from sklearn.linear_model import LogisticRegression
import joblib
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--outdir", default="train")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()
    preproc_path = os.path.join(args.outdir, "preprocessor", "preprocessor.pkl")
    meta_path = os.path.join(os.path.dirname(preproc_path), "preprocessor_meta.json")

    model_name = "logreg"

    X_train, X_test, y_train, y_test, feature_names, preproc = load_binary_split(
        args.csv, preproc_path, meta_path
    )

    model = LogisticRegression(
        C=args.C, max_iter=1000, n_jobs=-1, class_weight="balanced", solver="lbfgs"
    )

    print(f"[logreg] training ({X_train.shape[0]} samples, {X_train.shape[1]} features)...")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"[logreg] done in {time.time()-t0:.2f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    model_dir = os.path.join(args.outdir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    joblib.dump(preproc, os.path.join(model_dir, "preprocessor.pkl"))

    metrics = evaluate_model(
        y_test, y_pred, y_proba, feature_names, model,
        model_name=model_name, base_outdir=args.outdir
    )
    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[logreg] metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
