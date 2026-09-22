# IoT Device Network Audit — Binary & Multiclass Intrusion Detection System (IDS)

**Author:** Oleksandr Kuznetsov  
**Affiliation:** Università eCampus, Italy  
**License:** MIT  
**Version:** 0.1.0  

---

## 🔍 Overview

This repository provides an **end-to-end machine learning pipeline** for auditing IoT and IIoT network traffic security.  
It supports both **binary** (attack vs normal) and **multiclass** (attack type) classification tasks.

The system performs:
- structured preprocessing and feature encoding;
- exploratory data analysis and visualization;
- supervised training using modern ensemble models;
- quantitative comparison across accuracy, F1, ROC/PR AUC, and inference latency;
- per-class analysis for model explainability and reliability auditing;
- int8 quantization;
- multi-run benchmarking across multiple platforms;
- Mann-Whitney testing.

---

## 🧠 Research Context

IoT devices are among the most vulnerable elements of modern digital ecosystems.  
This project focuses on **auditable and interpretable IDS models** that can:
- detect common attack patterns in network flows (DoS, DDoS, MITM, password, ransomware, etc.),
- evaluate **robustness, latency, and resource footprint**, and  
- provide a baseline for **TinyML and federated IDS** deployments in smart environments.

The work is aligned with ongoing EU research directions in **edge security, federated analytics, and trustworthy AI**.

---

## ⚙️ Architecture Overview
```
src/
└── iot_audit/
├── preprocessing.py / preprocessing_mc.py # Data loading & encoding (binary / multiclass)
├── metrics.py / metrics_mc.py # Metrics, visualization, reports
scripts/
├── analyze_dataset.py, visualize_dataset.py # EDA, summary statistics, correlation plots
├── train_.py # Model training (RF, LGBM, XGB, LogReg)
├── compare_models.py # Binary comparison
├── train_mc_.py # Multiclass variants
├── compare_models_mc.py # Multiclass comparison and benchmarks
├── quantize_model.py # Binary float-to-int8 model quantization script
├── quantize_model_mc.py # Multiclass float-to-int8 model quantization script
└── run_mann-whitney.py # Mann-Whitney testing on lgbm/xgb latency ratio
reports*/ # Generated metrics, plots, and summaries
train*/ # Trained models
benchmark/ # Generated platform-specific benchmark data and charts
```

Each model is isolated under its own folder, ensuring reproducibility and traceability.

---

## 📊 Benchmark Summary (snapshot)

|Model   |Accuracy          |F1-Pos            |F1-Neg            |ROC-AUC           |PR-AUC            |FP  |FN  |Model Size MB|Preproc Size MB|Total Size MB|total_ms_per_1k avg Apple M1|
|--------|------------------|------------------|------------------|------------------|------------------|----|----|-------------|---------------|-------------|----------------------------|
|rf      |0.9988154185126394|0.9992239639919293|0.9974984990994596|0.9999939846005775|0.9999981112934438|31  |19  |23.853       |0.009          |23.861       |10.978537 ± 1.907783        |
|lgbm    |0.9992655594778365|0.9995187008026829|0.9984506971862662|0.9999795398801576|0.9999944628319596|11  |20  |2.749        |0.009          |2.758        |13.940153 ± 1.425839        |
|xgb     |0.9988864934018811|0.9992704015895931|0.9976498824941247|0.999977653761371 |0.9999936833597574|24  |23  |0.943        |0.009          |0.952        |6.114435 ± 0.224525         |
|logreg  |0.8769693667227368|0.9183606093477338|0.7504445191984238|0.9158887981620045|0.961069955424764 |2192|3001|0.004        |0.009          |0.013        |5.794473 ± 0.326560         |
|mlp     |0.9945035418986472|0.9964034353393483|0.9883487344314986|0.9993536278679872|0.99979374717221  |160 |72  |0.33         |0.009          |0.338        |4.313912 ± 1.748964         |
|mlp_int8|0.9709303703001729|0.9809516416983621|0.938659201119832 |0.9847192741159304|0.9945232333861591|612 |615 |0.028        |0.009          |0.037        |4.978423 ± 0.057768         |

Full benchmark results: https://github.com/acilgz/iot-audit/tree/main/benchmark

> All models were trained on the same dataset (`train_test_network.csv`, ~211k flows, 44 columns).  
> Metrics: stratified 80/20 split, consistent seed = 42.
> total_ms_per_1k: lower is better.

---

## Highlights
- Clean project layout: `src/`, `scripts/`, `reports*/` (artifacts), `models`, `benchmark/`.
- Reproducible preprocessing (imputation + one‑hot).
- Models: RandomForest, LightGBM, XGBoost, Logistic Regression, MLP (binary & multiclass).
- Metrics: accuracy, F1, ROC‑AUC/PR‑AUC, confusion matrix, per‑class report, lgbm/xgb latency ratio.
- Comparison scripts (quality, size, latency, Mann-Whitney); artifacts per model in isolated folders.
- Ready for GitHub CI: lint + basic import smoke.

## Dataset
Place your CSV (e.g., `data/train_test_network.csv`) with columns like:
```
src_ip,src_port,dst_ip,dst_port,proto,service,duration,src_bytes,dst_bytes,conn_state,...,label,type
```
- `label` → binary target (0=normal, 1=attack)
- `type`  → multiclass target (e.g., normal, ddos, dos, scanning, injection, xss, ransomware, password, backdoor, mitm)

> Note: heavy free‑text columns (e.g., `http_uri`, `ssl_subject`) are dropped by default.

## Quickstart

```bash
# 1) Create Python 3.12.14 venv
# sudo apt update && sudo apt install -y curl # Debian
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12.14 --python-preference only-managed .venv

source .venv/bin/activate  # Linux/Mac
# . .venv/Scripts/activate  # Windows PowerShell

# 2) Install deps
uv pip install -U pip
uv pip install -r requirements.txt

# 3) Put data
# data/train_test_network.csv

# 4) EDA
python scripts/analyze_dataset.py --csv data/train_test_network.csv --outdir train
python scripts/visualize_dataset.py --csv data/train_test_network.csv --outdir train/figures

# 5) Binary training
python scripts/prepare_preprocessor.py
python scripts/train_rf.py    --csv data/train_test_network.csv --outdir train
python scripts/train_lgbm.py  --csv data/train_test_network.csv --outdir train
python scripts/train_xgb.py   --csv data/train_test_network.csv --outdir train
python scripts/train_logreg.py --csv data/train_test_network.csv --outdir train
python scripts/train_mlp.py   --csv data/train_test_network.csv --outdir train
python scripts/quantize_model.py --input_dir train/models/mlp --csv data/train_test_network.csv

# 6) Binary comparison
python scripts/compare_models.py --outdir benchmark/sys1/binary --models-dir train --models rf lgbm xgb logreg mlp mlp_int8 --benchmark --sample_size 10000

# 7) Multiclass training
python scripts/prepare_preprocessor_mc.py
python scripts/train_mc_rf.py    --csv data/train_test_network.csv --outdir train_mc
python scripts/train_mc_lgbm.py  --csv data/train_test_network.csv --outdir train_mc
python scripts/train_mc_xgb.py   --csv data/train_test_network.csv --outdir train_mc
python scripts/train_mc_logreg.py --csv data/train_test_network.csv --outdir train_mc
python scripts/train_mc_mlp.py   --csv data/train_test_network.csv --outdir train_mc
python scripts/quantize_model_mc.py --input_dir train_mc/models/mlp_mc --csv data/train_test_network.csv

# 8) Multiclass comparison
python scripts/compare_models_mc.py --outdir benchmark/sys1/multiclass --models-dir train_mc --models rf_mc lgbm_mc xgb_mc logreg_mc mlp_mc mlp_mc_int8 --benchmark --sample_size 10000

# 9) Mann-Whitney lgbm/xgb
python scripts/run_mann-whitney.py --benchmark benchmark --aarch64 bcm2712,apple_m1,apple_m5 --x64 corei7_3770,corei5_7200U,ryzen7_7700
```

## Artifact layout

```
  train/
  models/          # Moved model artifacts here (instead of inside reports/)
    rf|lgbm|xgb|logreg/
      model.pkl
      preprocessor.pkl
      metrics.json
      leakage_report.json
      feature_importances.csv
  figures/
    rf|lgbm|xgb|logreg|mlp/
      roc_curve.png, pr_curve.png, confusion_matrix.png, feature_importances_top30.png
  summary/
    charts/        # New subfolder for all generated charts and plots
      accuracy.png
      f1_pos.png
      roc_auc.png
      fp.png
      fn.png
      latency_total_ms_per_1k.png
    summary_models.csv
    inference_benchmark*.csv
    lgbm_xgb_ratios_raw.csv

  train_mc/
  models/          # Moved model artifacts here (instead of inside reports_mc/)
    <model>_mc/
      model.pkl, preprocessor.pkl, metrics.json, per_class_report.csv, label_map.json, feature_importances.csv
  figures/
    <model>_mc/
      confusion_matrix.png, pr_micro.png, pr_<k>_<class>.png, feature_importances_top30.png
  summary/
    charts/        # New subfolder for all generated charts and plots
      accuracy.png
      macro_f1.png
      weighted_f1.png
      roc_auc_micro.png
      roc_auc_macro.png
      pr_auc_micro.png
      pr_auc_macro.png
      total_size_mb.png
      per_class_f1_*.png
    summary_models_mc.csv
    per_class_report_merged.csv
    inference_benchmark_mc*.csv
```

## Reproducibility & Notes
- Stratified split (80/20).
- Known leakage columns are dropped (e.g., `type` in binary task).
- Probabilities used to compute ROC/PR curves; safe handling if unavailable.
- For fair comparison, use the same CSV and seeds.

## Results (example snapshot)
- **LGBM‑MC**: accuracy ~0.9903, macro‑F1 ~0.9694, ROC‑AUC micro ~0.99994.
- **RF‑MC**: accuracy ~0.9897, macro‑F1 ~0.9681 (close to LGBM‑MC).
- Inference latency per 1k flows (10k sample): `logreg_mc` ~**5.44ms**, `lgbm_mc` ~**45.39ms**.

> Adjust thresholds for risk appetite: minimize FP for production or maximize recall on critical classes.

## 📚 Dataset: TON_IoT Network Dataset

**Name:** TON_IoT Network Dataset — IoT/IIoT network traffic for intrusion detection  
**Provider:** Cyber Range & IoT Labs, UNSW Canberra (SEIT) — *TON_IoT dataset collection*  
**Official page:** https://research.unsw.edu.au/projects/toniot-datasets  
**License:** Creative Commons **Attribution 4.0 International (CC BY 4.0)** (see the TON_IoT site for details)

This repository uses the *train/test* network flows subset often distributed as
`train_test_network.csv` (~29.9 MB; 44 columns). The flows were captured in realistic
IoT/IIoT smart-environment scenarios using tools such as **Argus** and **Bro (Zeek)**.
The dataset contains **benign** and **malicious** traffic and is suitable for
intrusion detection, anomaly detection, and ML benchmarking.

> **Columns (examples, 10 of 44):**
> `src_ip, src_port, dst_ip, dst_port, proto, service, duration, src_bytes, dst_bytes, conn_state, ...`
>
> Targets used in this repo:
> - `label` — binary (0 = normal, 1 = attack)  
> - `type`  — multiclass (e.g., `normal, ddos, dos, scanning, injection, xss, ransomware, password, backdoor, mitm`)

**Notes & caveats**
- Some high-cardinality text fields (e.g., `http_uri`, `ssl_subject`, `ssl_issuer`) are dropped by default to avoid leakage and reduce sparsity.
- Distribution is imbalanced across classes (e.g., `mitm` is rare). We report **macro-F1** and per-class metrics.
- Source of the CSV used here: community mirror (e.g., Kaggle: *ToN_IoT Network Dataset* https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset). Refer to the **official UNSW page** for canonical downloads and documentation.

### 📝 Acknowledgments
We gratefully acknowledge **The TON_IoT Datasets** team at **UNSW Canberra** for creating and maintaining the dataset collection. Free academic use is permitted under CC BY 4.0; for commercial use consult the dataset authors.

## How to contribute
See [CONTRIBUTING.md](CONTRIBUTING.md). Please run lint before commits.

## Citation
See [CITATION.cff](CITATION.cff).

## License
[MIT](LICENSE)
