from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from iot_audit.preprocessing import load_and_prepare_data
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/train_test_network.csv")
    ap.add_argument("--outdir", default="train")
    args = ap.parse_args()
    preproc_path = os.path.join(args.outdir, "preprocessor", "preprocessor.pkl")
    meta_path = os.path.join(os.path.dirname(preproc_path), "preprocessor_meta.json")

    _, _, _, _, feature_names, preproc = load_and_prepare_data(
        csv_path=args.csv, target_col="label", test_size=0.2, random_state=42,
        leakage_base=args.outdir, model_name="../preprocessor"
    )

    os.makedirs(os.path.dirname(preproc_path) or ".", exist_ok=True)
    joblib.dump(preproc, preproc_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)

    print(f"[prepare] preprocessor saved to {preproc_path}")
    print(f"[prepare] metadata ({len(feature_names)} features) saved to {meta_path}")

if __name__ == "__main__":
    main()
