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
benchmark/ # Generated platform-specific benchmark data and charts
runs/*/ # Trained models and logs per run
```

Each model is isolated under its own folder, ensuring reproducibility and traceability.

---

## 📊 Benchmark Summary (run '007')

|Model|Accuracy|F1-Pos|F1-Neg|ROC-AUC|PR-AUC|FP|FN|Model Size (MB)|Preproc Size (MB)|Total Size (MB)|Apple M1 (1)|Apple M5 (1)|BCM2712 (1)|Core i5-7200U (1)|Core i7-3770 (1)|Ryzen 7 7700 (1)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|rf|0.998839|0.999239|0.997548|0.999994|0.999998|31|18|23.957|0.009|23.966|7.502 ± 0.396|5.098 ± 0.207|23.305 ± 0.861|19.725 ± 0.300|13.209 ± 0.429|5.713 ± 0.168|
|lgbm|0.999266|0.999519|0.998451|0.999980|0.999994|11|20|2.749|0.009|2.758|10.014 ± 0.133|6.060 ± 0.665|31.517 ± 0.412|18.618 ± 0.033|11.023 ± 0.043|3.745 ± 0.019|
|xgb|0.998886|0.999270|0.997650|0.999979|0.999994|24|23|0.942|0.009|0.951|3.536 ± 0.089|2.207 ± 0.059|10.384 ± 0.191|10.194 ± 0.021|7.362 ± 0.016|2.885 ± 0.007|
|logreg|0.957331|0.972033|0.910044|0.992810|0.997878|890|911|0.004|0.009|0.013|2.688 ± 0.018|1.611 ± 0.009|7.776 ± 0.022|12.671 ± 0.192|9.344 ± 0.007|3.679 ± 0.044|
|mlp|0.994930|0.996682|0.989254|0.999399|0.999803|150|64|0.327|0.009|0.335|3.054 ± 0.035|1.931 ± 0.020|10.132 ± 0.129|8.342 ± 0.290|7.084 ± 0.288|3.042 ± 0.238|
|mlp_int8|0.987159|0.991555|0.973221|0.998651|0.999301|151|391|0.028|0.009|0.037|2.893 ± 0.149|1.767 ± 0.088|8.898 ± 0.067|8.154 ± 0.172|7.258 ± 0.208|2.684 ± 0.043|

(1) ms/1k, mean ± std. deviation

Full benchmark results [here](benchmark)

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
source "$HOME/.local/bin/env"
uv venv --python 3.12.14 --python-preference only-managed .venv

source ".venv/bin/activate"  # Linux/Mac

# 2) Install deps
uv pip install -U pip
uv pip install -r requirements.txt

# 3) Put data
# data/train_test_network.csv

# 4) Train once per run
bash train.sh 001

# 5) Benchmark once per machine
bash benchmark.sh 001 apple_m5

# 6) Mann-Whitney lgbm/xgb
python scripts/run_mann-whitney.py --benchmark benchmark/001 --aarch64 bcm2712,apple_m1,apple_m5 --x64 corei7_3770,corei5_7200U,ryzen7_7700

# 7) Plot platform comparison bars
python scripts/benchmark_bars.py --style yerr --input-dir benchmark/001 --output-dir benchmark/001/charts
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

## Results (example run '007')
- **LGBM-MC:** accuracy 98.95%, macro-F1 0.9662.
- **RF-MC:** accuracy 98.87%, macro-F1 0.9651.
- **MLP-MC FP32:** accuracy 95.21%, macro-F1 0.8995.
- **MLP-MC INT8:** accuracy 66.78%, macro-F1 0.6057.
- **MLP-MC input diagnostics:** Keras FP32 94.91%; quantized-dequantized inputs 66.73%; restoring numeric features 92.20%; restoring one-hot features 69.26%.
- **MLP-MC duplicates:** 6431 of 42209 test rows (15.24%) share the selected raw input values with the effective fit partition.
- **MLP-MC INT8 per-class recall:** class indices and names in [`007/multiclass/models/mlp_mc_int8/per_class_report.csv`](runs/007/multiclass/models/mlp_mc_int8/per_class_report.csv) and [`007/multiclass/models/mlp_mc/label_map.json`](runs/007/multiclass/models/mlp_mc/label_map.json).

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
