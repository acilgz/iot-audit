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

|Model|Accuracy|F1-Pos|F1-Neg|ROC-AUC|PR-AUC|FP|FN|Model Size (MiB)|Preproc Size (MiB)|Model + Preproc (MiB)|Apple M1 (1)|Apple M5 (1)|BCM2712 (1)|Core i5-7200U (1)|Core i7-3770 (1)|Ryzen 7 7700 (1)|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|rf|0.998839|0.999239|0.997548|0.999994|0.999998|31|18|23.957|0.009|23.966|7.502 ± 0.396|5.098 ± 0.207|23.305 ± 0.861|19.725 ± 0.300|13.209 ± 0.429|5.713 ± 0.168|
|lgbm|0.999266|0.999519|0.998451|0.999980|0.999994|11|20|2.749|0.009|2.758|10.014 ± 0.133|6.060 ± 0.665|31.517 ± 0.412|18.618 ± 0.033|11.023 ± 0.043|3.745 ± 0.019|
|xgb|0.998886|0.999270|0.997650|0.999979|0.999994|24|23|0.942|0.009|0.951|3.536 ± 0.089|2.207 ± 0.059|10.384 ± 0.191|10.194 ± 0.021|7.362 ± 0.016|2.885 ± 0.007|
|logreg|0.957331|0.972033|0.910044|0.992810|0.997878|890|911|0.004|0.009|0.013|2.688 ± 0.018|1.611 ± 0.009|7.776 ± 0.022|12.671 ± 0.192|9.344 ± 0.007|3.679 ± 0.044|
|mlp|0.994930|0.996682|0.989254|0.999399|0.999803|150|64|0.327|0.009|0.335|3.054 ± 0.035|1.931 ± 0.020|10.132 ± 0.129|8.342 ± 0.290|7.084 ± 0.288|3.042 ± 0.238|
|mlp_int8|0.987159|0.991555|0.973221|0.998651|0.999301|151|391|0.028|0.009|0.037|2.893 ± 0.149|1.767 ± 0.088|8.898 ± 0.067|8.154 ± 0.172|7.258 ± 0.208|2.684 ± 0.043|

(1) Warm batch processing time: preprocessing + prediction, in **ms/1k flows**, measured on a batch of 10,000 rows sampled from the **full CSV** (seed 42).
Values are mean ± population SD (`ddof=0`) of five repeats on the same device.
Each physical device is one independent observation; repeats are averaged and do not represent additional devices.
These measurements do not include all the elements of a production IDS such as packet capture, feature extraction from packets or queueing and do not establish end-to-end, single-flow real-time latency.

File sizes are in **MiB (1,048,576 bytes)**: `model.pkl`, `model.keras` or `model.tflite`, plus `preprocessor.pkl`. The total excludes the separate MLP `scaler.pkl`, metadata and runtime memory. It does not represent RAM utilization. Each total is rounded after adding the original byte sizes.

In both binary and multiclass, MLP INT8 has lower mean total time than MLP FP32 on five of the six tested devices (FP32 is faster on the Core i7-3770).    
[Run 007 benchmark CSVs and manifests](benchmark/007) link timings to model hashes.     
[Run 001](benchmark/001) is a separate historical comparison.    
[Complete run data for 001 and 007 on the Releases page](../../releases)      

## Evaluation protocol

The dataset contains 211,043 rows. `label` and `type` are removed from input X before training in order to prevent data leakage. The appropriate target is supplied separately as Y.    
The stratified outer split (seed 42) contains **168,834 training rows and 42,209 test rows**.    
Within each classification type, all models use the recorded outer test indices; binary and multiclass stratification can produce different indices.

| Pipeline | Rows used to fit the classifier | Rows used to fit preprocessing | Numerical scaling |
|---|---:|---:|---|
| RF, LGBM, XGB | 168,834 outer-training rows | 135,067-row internal subset | None |
| LogReg | 168,834 outer-training rows | 135,067-row internal subset | StandardScaler inside preprocessing, fitted on that subset |
| MLP FP32 | 135,067 internal-fit rows; 33,767 validation rows for early stopping | 135,067 internal-fit rows | Separate scaler on numerical features only, fitted on internal fit |
| MLP INT8 | Converted from the corresponding MLP FP32 | Reuses its preprocessing and scaler | Same transformations; calibration uses 1,000 internal-fit rows |

Feature selection, imputation and one-hot encoding are learned on the recorded preprocessing-fit subset before transforming the other partitions.    
Only MLP holds the 33,767 rows out of classifier fitting for validation.    
The other classifiers are also trained on those rows. One-hot outputs remain binary (0/1) before INT8 input quantization.

| Purpose | Evaluation sample |
|---|---|
| Final accuracy, F1, ROC/PR and per-class results | Entire outer test: 42,209 rows |
| INT8 diagnosis and runtime/batch parity | Same first 10,000 rows in the recorded MLP internal-validation order |
| Cross-device timing | 10,000 rows sampled from the complete CSV, seed 42; may overlap training/test |

The diagnostic sample and timing sample have equal sizes but different origins and purposes.

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
# cp "$HOME/Downloads/train_test_network.csv" data/train_test_network.csv

# 4) Choose a new, unused run ID
export RUN_ID="010"

# 5) Train once per run
bash train.sh ${RUN_ID}

# 6) Benchmark once per machine
bash benchmark.sh ${RUN_ID} apple_m5

# 7) Mann-Whitney lgbm/xgb
python scripts/run_mann-whitney.py --benchmark benchmark/${RUN_ID} --aarch64 bcm2712,apple_m1,apple_m5 --x64 corei7_3770,corei5_7200U,ryzen7_7700

# 8) Plot platform comparison bars
python scripts/benchmark_bars.py --style yerr --input-dir benchmark/${RUN_ID} --output-dir benchmark/${RUN_ID}/charts
```

## Artifact layout

```text
runs/<run_id>/
  commit.txt, git-status.txt, environment.txt, python-version.txt
  <binary|multiclass>/models/<model>/
    model_manifest.json, preprocessor_meta.json, metrics.json
    model.pkl* | model.keras* | model.tflite*
    preprocessor.pkl*, scaler.pkl* (MLP only)
    per_class_report.csv, label_map.json
  audit/duplicate-audit.json, input-groups.json
  int8_diagnostics/runtime_batch_comparison.json
  logs/
  benchmark-logs/<device>/
benchmark/<run_id>/<device>/<binary|multiclass>/
  inference_benchmark{,_mc}.csv
  inference_benchmark{,_mc}.manifest.json
  summary/summary_models{,_mc}.csv
benchmark/<run_id>/charts/
```

\* Binary model files are excluded by .gitignore.

Run 007 data:    
[Training commit](runs/007/commit.txt)    
[Environment](runs/007/environment.txt)    
[Execution logs](runs/007/logs)    

Complete data for runs 001 and 007, including the models, preprocessor, scaler, and hashes, are available in [Releases](../../releases).

## Results: run 007, outer test

Multiclass results on 42,209 test rows.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| lgbm_mc | 98.946% | 0.96620 |
| rf_mc | 98.870% | 0.96506 |
| xgb_mc | 98.932% | 0.96514 |
| logreg_mc | 82.584% | 0.76799 |
| mlp_mc | 95.207% | 0.89949 |
| mlp_mc_int8 | 66.782% | 0.60574 |

Run 001 is kept as a historical baseline: multiclass FP32 95.193% / 0.89823 macro-F1; INT8 48.800% / 0.31760. Run 007 improves INT8 accuracy and macro-F1, while the gap from FP32 remains substantial. The history and corrections do not identify the isolated causal effect of changing one-hot scaling.

### INT8 diagnosis: validation, 10,000 rows

Keras FP32: **94.91%**    
Keras with quantized/dequantized inputs: **66.73%***    
INT8: **66.93%**.    
Keras with quantized/dequantized inputs, restoring only numerical features: **92.20%***    
Keras with quantized/dequantized inputs, restoring only one-hot features: **69.26%***     

\* These results are for diagnostic purposes only; they do not represent a deployable INT8 model.   

The loss in numerical-input resolution accounts for most of the observed loss: **129,921 of 160,000 numerical values (81.20%)** were non-zero before quantization and were rounded to zero after dequantization. Only **5 values** were clipped. The [input-group report](runs/007/audit/input-groups.json) records feature-level errors. The [runtime report](runs/007/int8_diagnostics/runtime_batch_comparison.json) records identical probabilities and classes for TF Lite/LiteRT with batch 1/10,000 on the same validation observations.    

[INT8 per-class recall](runs/007/multiclass/models/mlp_mc_int8/per_class_report.csv) is linked to [class indices and names](runs/007/multiclass/models/mlp_mc/label_map.json).

### Duplicate inputs and limits

The [duplicate audit](runs/007/audit/duplicate-audit.json) compares the 32 selected raw inputs, before imputation/OHE and without targets. Rows with at least one matching input in the other partition are counted; not pairs.

| Comparison | Binary | Multiclass |
|---|---:|---:|
| Outer train → test | 6,869 / 42,209 (16.2738%) | 6,963 / 42,209 (16.4965%) |
| MLP effective fit → test | 6,331 / 42,209 (14.9992%) | 6,431 / 42,209 (15.2361%) |
| MLP effective fit → validation | 5,091 / 33,767 (15.0769%) | 5,189 / 33,767 (15.3671%) |

The fit-based figures refer to MLP fitting, not to the full classifier-training set of every model.    
Input repetition limits the generalization measured by the random split. It is distinct from the corrected use of target-derived input columns.    

This study uses one dataset and one training seed, without a chronological split or testing on a secondary, independent dataset.

### Hardware comparison

Mann–Whitney compares one LGBM/XGB mean-total-time ratio per device: three aarch64 systems versus three x86_64 systems, separately for binary and multiclass. Both tasks give **U = 9, exact one-sided p = 0.05**, for the direction **larger ratios on the tested aarch64 systems**. See the [binary](benchmark/007/lgbm_xgb_mann-whitney.csv) and [multiclass](benchmark/007/lgbm_xgb_mann-whitney_mc.csv) reports.
Benchmark data for every system in each class (aarch64 and x86_64) must be collected before running `run_mann-whitney.py`.   

N.B.: This is an exploratory comparison of six complete systems, including other factors such as the OS/software, other hardware differences, and thermal constraints. It does not isolate the effect of the ISA.

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
