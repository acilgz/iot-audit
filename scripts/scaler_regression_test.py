import math, sys, joblib, pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    df = pd.read_csv("data/train_test_network.csv")
    n_train = len(df) - math.ceil(0.2 * len(df))  # same split as train_test_split(test_size=0.2)
    failed = False

    for base, names in [("reports", ["logreg", "rf", "xgb", "lgbm"]),
                        ("reports_mc", ["logreg_mc", "rf_mc", "xgb_mc", "lgbm_mc"])]:
        for name in names:
            p = joblib.load(f"{base}/models/{name}/preprocessor.pkl")
            m = joblib.load(f"{base}/models/{name}/model.pkl")
            scaler = p.named_transformers_["num"].named_steps["scaler"]
            cols = [c for n, _, cs in p.transformers_ if n != "remainder" for c in cs]
            checks = {
                "scaling matches map": isinstance(scaler, StandardScaler) == name.startswith("logreg"),
                "scaler fit on train only": not isinstance(scaler, StandardScaler) or scaler.n_samples_seen_ == n_train,
                "no target-derived cols": not {"label", "type"} & set(cols),
                "same pipeline at inference": len(m.predict(p.transform(df.head(100)))) == 100,
            }
            for k, v in checks.items():
                print(f"{'PASS' if v else 'ERROR'} [{name}] {k}")
                failed |= not v

    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
