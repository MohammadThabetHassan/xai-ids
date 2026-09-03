# XAI-IDS: Explainable AI Intrusion Detection System

[![CI](https://github.com/MohammadThabetHassan/xai-ids/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammadThabetHassan/xai-ids/actions/workflows/ci.yml)
[![Pages](https://github.com/MohammadThabetHassan/xai-ids/actions/workflows/pages.yml/badge.svg)](https://mohammadthabethassan.github.io/xai-ids/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Datasets](https://img.shields.io/badge/datasets-CICIDS2017%20%7C%20UNSW--NB15%20%7C%20CICIDS2018-orange.svg)](RESULTS.md)
[![Models](https://img.shields.io/badge/models-XGBoost%20%7C%20RF%20%7C%20LightGBM%20%7C%20Ensemble-purple.svg)](RESULTS.md)

An Explainable AI-based Intrusion Detection System that combines machine learning classifiers with **SHAP** and **LIME** explainability techniques, evaluated across **3 datasets** (CIC-IDS-2017, UNSW-NB15, CSE-CIC-IDS-2018) with a novel **XAI Confidence Score (XCS)** for measuring explanation reliability.

> **Abstract:** We present XAI-IDS, a multi-dataset intrusion detection framework combining XGBoost, Random Forest, LightGBM, and VotingEnsemble classifiers with SHAP and LIME explainability methods and a novel XAI Confidence Score (XCS) that quantifies explanation trustworthiness. Evaluated on 3 datasets totalling 16M+ records across 26 attack classes, our best model achieves F1=0.9964 on CIC-IDS-2017 and F1=0.7933 on UNSW-NB15. On synthetic probe sets (n=100 per dataset) the full three-term XCS flags 5% of UNSW-NB15 predictions for analyst review; those probe sets carry no ground-truth labels, so this is a flag rate and not a measure of flagging accuracy. Statistical significance testing (McNemar's test, paired t-test) validates model comparisons, and adversarial robustness evaluation demonstrates XCS degradation under FGSM attacks.

**Live Documentation:** [GitHub Pages](https://mohammadthabethassan.github.io/xai-ids/) | [Full Results](RESULTS.md) | [Model Card](MODEL_CARD.md)

---

## Table of Contents

- [Architecture](#architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Results](#results)
- [Explainability & XCS](#explainability--xcs)
- [API & Deployment](#api--deployment)
- [Repository Structure](#repository-structure)
- [Testing](#testing)
- [Limitations](#limitations)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Architecture

![XAI-IDS Architecture](xai_ids_Architecture.svg)

## Pipeline

![XAI-IDS Pipeline](xai_ids_pipeline.svg)

---

## Datasets

This project evaluates on **3 benchmark intrusion detection datasets**:

| Dataset | Classes | Features | Records | Type |
|---------|---------|----------|---------|------|
| **CIC-IDS-2017** | 14 (1 benign + 13 attacks) | 20 selected | 2.8M+ | Multi-class |
| **UNSW-NB15** | 10 (1 normal + 9 attacks) | 20 selected | 2.5M+ | Multi-class |
| **CSE-CIC-IDS-2018** | 2 captured (14+ available) | 20 selected | 10M+ | Multi-class (stratified fix applied) |

### CIC-IDS-2017 Attack Types

| Class | Description |
|-------|-------------|
| BENIGN | Normal network traffic |
| DoS Hulk | HTTP flood denial-of-service |
| PortScan | Network port scanning |
| DDoS | Distributed denial-of-service |
| DoS GoldenEye | HTTP DoS using GoldenEye tool |
| FTP-Patator | FTP brute force |
| SSH-Patator | SSH brute force |
| DoS slowloris | Slowloris DoS attack |
| DoS Slowhttptest | Slow HTTP DoS attack |
| Bot | Botnet traffic |
| Web Attack - Brute Force | Web application brute force |
| Web Attack - XSS | Cross-site scripting |
| Infiltration | Network infiltration |
| Web Attack - Sql Injection | SQL injection |

**Sources:**
- CIC-IDS-2017: [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) | [Zenodo Mirror](https://zenodo.org/records/10141593)
- UNSW-NB15: [UNSW Canberra](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- CSE-CIC-IDS-2018: [CSE-CIC](https://www.unb.ca/cic/datasets/ids-2018.html)

---

## Installation

### Prerequisites

- Python 3.10+
- pip
- (Optional) Docker for containerized execution

### Install

```bash
# Clone the repository
git clone https://github.com/MohammadThabetHassan/xai-ids.git
cd xai-ids

# Install dependencies
pip install -r requirements.txt
```

### Docker

```bash
# Build the container
docker build -t xai-ids .

# Run the pipeline
docker run -v $(pwd)/outputs:/app/outputs xai-ids
```

---

## Quick Start

### Full Pipeline (Synthetic Data)

```bash
python run_pipeline.py
```

### Download Real CIC-IDS-2017 Dataset

```bash
python run_pipeline.py --download
```

### Quick Test (Small Dataset)

```bash
python run_pipeline.py --sample-size 5000 --skip-explain
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--download` | Download real CIC-IDS-2017 data (via Zenodo fallback) |
| `--sample-size N` | Synthetic dataset size (default: 50000) |
| `--random-sample` | Use random sampling instead of first N rows |
| `--skip-explain` | Skip SHAP and LIME analysis |
| `--skip-shap` | Skip SHAP only |
| `--skip-lime` | Skip LIME only |
| `--models lr rf xgb` | Select specific models |
| `--shap-samples N` | Samples for SHAP (default: 500) |
| `--cv-folds N` | Run N-fold cross-validation (0 to skip) |
| `--pr-curves` | Generate precision-recall curves |
| `--calibration` | Generate calibration curves |
| `--failure-analysis` | Generate failure analysis report |

### Using Make

```bash
make pipeline          # Full pipeline
make pipeline-small    # Quick test with small data
make pipeline-fast     # Skip explainability
make test              # Run tests
make clean             # Remove outputs
```

---

## Results

### Multi-Dataset Performance (Kaggle Notebook)

Results from the multi-dataset evaluation with XGBoost, Random Forest, LightGBM, and VotingEnsemble.

#### CIC-IDS-2017 (14 classes)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **0.9966** | **0.9964** | **0.9966** | **0.9964** |
| VotingEnsemble | 0.9886 | 0.9945 | 0.9886 | 0.9911 |
| RandomForest | 0.9857 | 0.9940 | 0.9857 | 0.9893 |
| LightGBM | 0.9744 | 0.9930 | 0.9744 | 0.9828 |

#### UNSW-NB15 (10 classes)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **0.8004** | **0.8120** | **0.8004** | **0.7982** |
| VotingEnsemble | 0.7867 | 0.8240 | 0.7867 | 0.8002 |
| RandomForest | 0.7635 | 0.8202 | 0.7635 | 0.7848 |
| LightGBM | 0.7630 | 0.8291 | 0.7630 | 0.7863 |

#### CSE-CIC-IDS-2018 (2 classes captured — full multi-class results pending)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **RandomForest** | **0.9993** | **0.9993** | **0.9993** | **0.9992** |
| VotingEnsemble | 0.9993 | 0.9993 | 0.9993 | 0.9992 |
| XGBoost | 0.9990 | 0.9990 | 0.9990 | 0.9990 |
| LightGBM | 0.9990 | 0.9990 | 0.9990 | 0.9990 |

> **See [RESULTS.md](RESULTS.md) for complete per-class breakdowns, XCS scores, and cross-dataset analysis.**

### Multi-Dataset Visualizations

#### CIC-IDS-2017 Confusion Matrix
![CIC-IDS-2017 Confusion](plots/confusion_CICIDS2017.png)

#### UNSW-NB15 Confusion Matrix
![UNSW-NB15 Confusion](plots/confusion_UNSWNB15.png)

#### CSE-CIC-IDS-2018 Confusion Matrix
![CSE-CIC-IDS-2018 Confusion](plots/confusion_CICIDS2018.png)

#### SHAP Beeswarm Plots
![CIC-IDS-2017 SHAP](plots/shap_beeswarm_CICIDS2017.png)
![UNSW-NB15 SHAP](plots/shap_beeswarm_UNSWNB15.png)
![CSE-CIC-IDS-2018 SHAP](plots/shap_beeswarm_CICIDS2018.png)

#### XCS Distribution
![CIC-IDS-2017 XCS](plots/xcs_CICIDS2017.png)
![UNSW-NB15 XCS](plots/xcs_UNSWNB15.png)
![CSE-CIC-IDS-2018 XCS](plots/xcs_CICIDS2018.png)

#### Cross-Dataset Comparison
![Cross-Dataset](plots/cross_dataset_comparison.png)

#### LIME Local Explanations
![LIME Benign](plots/lime_CICIDS2017_Benign.png)
![LIME DDoS](plots/lime_CICIDS2017_DDoS.png)

#### SHAP Waterfall Plots
![SHAP Waterfall Benign](plots/shap_waterfall_CICIDS2017_Benign.png)
![SHAP Waterfall Bot](plots/shap_waterfall_CICIDS2017_Bot.png)
![SHAP Waterfall DDoS](plots/shap_waterfall_CICIDS2017_DDoS.png)

### Synthetic Data Results (Main Pipeline)

| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) |
|-------|----------|-----------|--------|---------------|------------|
| **XGBoost** | **0.8250** | **0.8158** | **0.8250** | **0.8193** | **0.51** |
| Random Forest | 0.8347 | 0.8096 | 0.8347 | 0.8162 | 0.48 |
| Logistic Regression | 0.7601 | 0.7866 | 0.7601 | 0.7651 | 0.48 |

### Model Comparison Chart

![Model Comparison](outputs/figures/model_comparison.png)

### Confusion Matrices

#### Logistic Regression
![LR Confusion Matrix](outputs/figures/confusion_matrix_logistic_regression_normalized.png)

#### Random Forest
![RF Confusion Matrix](outputs/figures/confusion_matrix_random_forest_normalized.png)

#### XGBoost
![XGB Confusion Matrix](outputs/figures/confusion_matrix_xgboost_normalized.png)

---

## Explainability & XCS

### SHAP - Global Feature Importance

SHAP (SHapley Additive exPlanations) provides global feature importance by computing the average contribution of each feature to predictions.

#### Random Forest - SHAP Feature Importance
![RF SHAP](outputs/figures/shap_feature_importance_random_forest.png)

#### XGBoost - SHAP Feature Importance
![XGB SHAP](outputs/figures/shap_feature_importance_xgboost.png)

### LIME - Local Explanations

LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by fitting surrogate models around specific data points.

#### Correctly Classified Sample (Random Forest)
![RF LIME Correct](outputs/figures/lime_correct_random_forest.png)

#### Misclassified Sample (XGBoost)
![XGB LIME Misclassified](outputs/figures/lime_misclassified_xgboost.png)

### XAI Confidence Score (XCS)

The **XCS** is a novel metric that measures the reliability of explanations for individual predictions:

```
XCS = w1 × Confidence + w2 × (1 - SHAP_Instability) + w3 × Jaccard(SHAP, LIME)
```

Where:
- **w1 = 0.4**: Weight for model prediction confidence
- **w2 = 0.3**: Weight for SHAP explanation stability (1 - feature ranking variance)
- **w3 = 0.3**: Weight for agreement between SHAP and LIME (Jaccard similarity of top-k features)

**Interpretation:**
- XCS > 0.7: High-confidence, reliable explanation
- XCS 0.3–0.7: Moderate confidence, use with caution
- XCS < 0.3: Low-confidence explanation, do not trust

#### XCS Results Summary (real data — Kaggle run)

| Dataset | Mean XCS | Correct XCS | Wrong XCS | Flagged (< 0.3) |
|---------|----------|-------------|-----------|-----------------|
| CIC-IDS-2017 | 0.517 | 0.526 | 0.469 | 0 / 74 (0%) |
| UNSW-NB15 | 0.424 | 0.469 | 0.360 | 14 / 80 (17.5%) |
| CSE-CIC-IDS-2018 | 0.550 | 0.550 | N/A | 0 / 86 (0%) |

Key finding: wrong predictions score lower than correct ones across all
datasets. This separation cannot yet be credited to the explainability terms.
In this run `shap_instability` is 0.5 and `jaccard_sl` is 0.0 for every
sample, because per-sample LIME did not execute and the instability ratio
saturated its 0.5 clamp. XCS therefore reduces to `0.4 x confidence + 0.15`
and correlates with confidence at r = 1.000 on CIC-IDS-2017 and UNSW-NB15
(0.998 on CSE-CIC-IDS-2018). The separation shown here is the well-known
tendency of a calibrated classifier to be less confident when wrong; it is
not independent evidence for the three-term metric.

The flag rate in this table (17.5% on UNSW-NB15) and the 5% quoted in the
abstract come from different runs and are not comparable: this table is the
real-data run with the two explainability terms inert, while the 5% figure
is the synthetic probe set where the full formula executes but no labels
exist. Establishing whether XCS improves on confidence alone requires
re-running the full formula against a labelled real test split and
comparing the two as review triggers.
Values computed on real data with full XCS formula (n=74–86 per dataset).
See `explanations/xcs_*.csv` for per-sample data.

![XCS Distribution](plots/xcs_CICIDS2017.png)

### SHAP vs LIME Agreement

The Jaccard similarity between SHAP and LIME top-k features measures how consistently both methods identify important features:

![Jaccard Similarity](plots/jaccard_shap_lime.png)

| Dataset | Mean Jaccard (SHAP–LIME top-5) |
|---------|-------------------------------|
| CIC-IDS-2017 | 0.180 |
| UNSW-NB15 | 0.111 |
| CSE-CIC-IDS-2018 | 0.429 |
| **Overall** | **0.216** |

Cross-dataset Jaccard similarity: **0.216** — indicating dataset-specific feature patterns.

---

## API & Deployment

### FastAPI Inference Server

```bash
# Start the API server
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0] * 78}'

# Get explanation with prediction
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0] * 78}'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and loaded models |
| `/health` | GET | Health check (models, scaler, encoder status) |
| `/predict` | POST | Make prediction with confidence and probabilities |
| `/explain` | POST | Prediction with top-10 feature importances |
| `/classes` | GET | List available prediction classes |
| `/xcs-summary` | GET | XCS offline evaluation summary per dataset |

### Docker

```bash
# Build
docker build -t xai-ids .

# Run pipeline
docker run -v $(pwd)/outputs:/app/outputs xai-ids

# Run API server
docker run -p 8000:8000 xai-ids uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

## Repository Structure

```
xai-ids/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # CI workflow (tests + imports)
│   │   └── pages.yml                 # GitHub Pages deployment
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
├── api/
│   └── app.py                        # FastAPI inference server
├── data/
│   ├── raw/                          # Raw CSV files
│   └── processed/                    # Processed data
├── docs/
│   ├── methodology.md                # Technical methodology
│   └── site/                         # GitHub Pages site
├── explanations/
│   ├── shap_lime_jaccard_all.csv     # SHAP-LIME agreement
│   ├── xcs_*.csv                     # XCS scores per dataset
│   └── *.html                        # LIME explanation files
├── models/                           # Trained models (Kaggle notebook)
├── notebooks/
│   └── exploration.ipynb             # Interactive demo notebook
├── outputs/
│   ├── figures/                      # Confusion matrices, SHAP, LIME plots
│   ├── models/                       # Trained models (.pkl) from pipeline
│   ├── reports/                      # Classification reports, failure analysis
│   └── results_metrics.csv           # Model comparison metrics
├── plots/                            # Multi-dataset visualizations
├── src/
│   ├── data/
│   │   ├── download.py               # Dataset download utility
│   │   ├── generate_sample.py        # Synthetic data generator
│   │   ├── loader.py                 # Chunked CSV loader
│   │   └── preprocessing.py          # Cleaning, encoding, scaling
│   ├── features/
│   │   └── feature_engineering.py    # Feature selection utilities
│   ├── models/
│   │   └── train.py                  # Model training (LR, RF, XGB)
│   ├── evaluation/
│   │   └── metrics.py                # Metrics, confusion matrices, charts
│   ├── explainability/
│   │   └── explain.py                # SHAP and LIME implementations
│   └── utils/
│       └── logger.py                 # Centralized logging
├── tests/
│   ├── test_smoke.py                 # Smoke tests (13 tests)
│   └── test_evaluation.py            # Evaluation tests (11 tests)
├── xai_ids_multidataset.ipynb        # Kaggle notebook (multi-dataset)
├── config.yaml                       # Reference configuration
├── Dockerfile                        # Container definition
├── Makefile                          # Build automation
├── MODEL_CARD.md                     # Model card (HuggingFace-style)
├── README.md                         # This file
├── RESULTS.md                        # Full results tables
├── requirements.txt                  # Python dependencies
└── run_pipeline.py                   # Main pipeline entry point
```

---

## Reproducibility

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the supported synthetic-data quick start, expected artifacts, validation commands, and real-dataset caveats.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run smoke tests only
python -m pytest tests/test_smoke.py -v -x
```

Tests include:
- Module import verification (8 tests)
- Synthetic data generation (2 tests)
- Data preprocessing (2 tests)
- Mini end-to-end pipeline execution (1 test)
- Metrics computation (2 tests)
- Cross-validation (2 tests)
- Calibration curves (1 test)
- Failure analysis (2 tests)
- Confusion matrix (2 tests)
- Edge cases (2 tests)

---

## Limitations

1. **Class Imbalance**: Severe imbalance in CIC-IDS-2017 (~600:1 ratio) causes poor detection of minority attack classes. Balanced class weights improve macro F1 from 0.44 to 0.51 but rare classes (SQL Injection, Heartbleed) remain challenging.

2. **Feature Selection**: The Kaggle notebook uses 20 features per dataset (selected by importance), while the main pipeline uses all 78 CIC-IDS-2017 features. Results are not directly comparable.

3. **Synthetic vs Real Data**: Default `run_pipeline.py` uses synthetic data. Real data requires download via `--download` or running the Kaggle notebook.

4. **Dataset Generalization**: Cross-dataset feature importance shows low Jaccard similarity (0.216), meaning models trained on one dataset may not generalize well to others.

5. **Research-Grade**: This is a research/educational system, not a production-ready IDS. It should be used as part of a defense-in-depth strategy.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{xai_ids_2026,
  author = {Hassan, Mohammad Thabet and Sadek, Fahad and Sami, Ahmed},
  title = {XAI-IDS: Explainable AI Intrusion Detection System},
  year = {2026},
  url = {https://github.com/MohammadThabetHassan/xai-ids},
  license = {MIT}
}
```

---

## Authors

| Name | Role | GitHub |
|------|------|--------|
| **Mohammad Thabet Hassan** | Lead Developer & Primary Contributor | [@MohammadThabetHassan](https://github.com/MohammadThabetHassan) |
| **Fahad Sadek** | Contributor | [@fahad6789123](https://github.com/fahad6789123) |
| **Ahmed Sami** | Contributor | [@AhmedSamiAlameri](https://github.com/AhmedSamiAlameri) |

**Supervisor:** Dr. Mehak Khurana — Mehak.Khurana@cud.ac.ae

---

## Acknowledgments

- [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/) — CIC-IDS-2017 and CSE-CIC-IDS-2018 datasets
- [UNSW Canberra Cyber](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/) — UNSW-NB15 dataset
- [SHAP](https://github.com/shap/shap) — Lundberg & Lee, NeurIPS 2017
- [LIME](https://github.com/marcotcr/lime) — Ribeiro, Singh & Guestrin, KDD 2016

---

## References

1. Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A., "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018.
2. Lundberg, S.M. and Lee, S.-I., "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
3. Ribeiro, M.T., Singh, S. and Guestrin, C., "Why Should I Trust You? Explaining the Predictions of Any Classifier," KDD 2016.
4. Moustafa, N. and Slay, J., "UNSW-NB15: a comprehensive data set for network intrusion detection systems," MilCIS 2015.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
