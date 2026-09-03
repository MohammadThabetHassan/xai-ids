# XAI-IDS Benchmarks

Benchmark results for every model and dataset configuration reported in this repository, with the exact run settings used to produce them.

---

## Synthetic Data Benchmarks (run_pipeline.py)

| Dataset | Model | Train Size | Test Size | Features | Accuracy | Precision (W) | Recall (W) | F1 (Weighted) | F1 (Macro) | Train Time (s) | Inference (ms/sample) |
|---------|-------|-----------|----------|----------|----------|--------------|-----------|-------------|-----------|---------------|---------------------|
| CIC-IDS-2017 (synthetic) | XGBoost | 35,000 | 10,000 | 78 | 0.8250 | 0.8158 | 0.8250 | 0.8193 | 0.51 | ~2.5 | ~0.3 |
| CIC-IDS-2017 (synthetic) | Random Forest | 35,000 | 10,000 | 78 | 0.8347 | 0.8096 | 0.8347 | 0.8162 | 0.48 | ~3.0 | ~0.5 |
| CIC-IDS-2017 (synthetic) | Logistic Regression | 35,000 | 10,000 | 78 | 0.7601 | 0.7866 | 0.7601 | 0.7651 | 0.48 | ~1.0 | ~0.1 |

## Multi-Dataset Benchmarks (Kaggle Notebook)

### CIC-IDS-2017 (14 classes, 20 selected features)

| Model | Train Size | Test Size | Accuracy | Precision (W) | Recall (W) | F1 (Weighted) | Train Time (s) |
|-------|-----------|----------|----------|--------------|-----------|-------------|---------------|
| XGBoost | ~2.8M | ~560K | 0.9966 | 0.9964 | 0.9966 | 0.9964 | ~120 |
| VotingEnsemble | ~2.8M | ~560K | 0.9886 | 0.9945 | 0.9886 | 0.9911 | ~180 |
| Random Forest | ~2.8M | ~560K | 0.9857 | 0.9940 | 0.9857 | 0.9893 | ~200 |
| LightGBM | ~2.8M | ~560K | 0.9744 | 0.9930 | 0.9744 | 0.9828 | ~60 |

### UNSW-NB15 (10 classes, 20 selected features)

| Model | Train Size | Test Size | Accuracy | Precision (W) | Recall (W) | F1 (Weighted) | Train Time (s) |
|-------|-----------|----------|----------|--------------|-----------|-------------|---------------|
| XGBoost | ~2.5M | ~500K | 0.8004 | 0.8120 | 0.8004 | 0.7982 | ~90 |
| VotingEnsemble | ~2.5M | ~500K | 0.7867 | 0.8240 | 0.7867 | 0.8002 | ~150 |
| Random Forest | ~2.5M | ~500K | 0.7635 | 0.8202 | 0.7635 | 0.7848 | ~180 |
| LightGBM | ~2.5M | ~500K | 0.7630 | 0.8291 | 0.7630 | 0.7863 | ~50 |

### CSE-CIC-IDS-2018 (2 classes, 20 selected features)

| Model | Train Size | Test Size | Accuracy | Precision (W) | Recall (W) | F1 (Weighted) | Train Time (s) |
|-------|-----------|----------|----------|--------------|-----------|-------------|---------------|
| XGBoost | ~10M | ~2M | 0.9990 | 0.9990 | 0.9990 | 0.9990 | ~150 |
| RandomForest | ~10M | ~2M | 0.9993 | 0.9993 | 0.9993 | 0.9992 | ~250 |
| LightGBM | ~10M | ~2M | 0.9990 | 0.9990 | 0.9990 | 0.9990 | ~70 |
| Ensemble | ~10M | ~2M | 0.9993 | 0.9993 | 0.9993 | 0.9992 | ~200 |

## XCS Benchmarks (real data — Kaggle run)

| Dataset | Model | Mean XCS | Correct XCS | Wrong XCS | Flagged (< 0.3) |
|---------|-------|----------|-------------|-----------|-----------------|
| CIC-IDS-2017 | XGBoost | 0.517 | 0.526 | 0.469 | 0 / 74 (0%) |
| UNSW-NB15 | XGBoost | 0.424 | 0.469 | 0.360 | 14 / 80 (17.5%) |
| CSE-CIC-IDS-2018 | XGBoost | 0.550 | 0.550 | N/A | 0 / 86 (0%) |

## API Latency Benchmarks

| Endpoint | Mean Latency (ms) | P95 Latency (ms) | Notes |
|----------|-------------------|------------------|-------|
| GET / | < 1 | < 5 | Health check |
| POST /predict | ~10 | ~20 | Fast-path XCS (confidence only) |
| POST /explain | ~300 | ~500 | Full XCS with SHAP + LIME |
| GET /xcs-summary | < 5 | < 10 | Reads pre-computed CSVs |
| GET /health/features | < 5 | < 10 | Returns scaler statistics |

## Statistical Significance

All pairwise model comparisons use McNemar's test (α = 0.05).
Results show XGBoost is significantly better than LightGBM on
CIC-IDS-2017 (p < 0.001) but not significantly different from
Random Forest on CIC-IDS-2018 (p > 0.05) due to ceiling effects.

## SHAP–LIME Agreement

| Dataset | Mean Jaccard (top-5 features) |
|---------|------------------------------|
| CIC-IDS-2017 | 0.180 |
| UNSW-NB15 | 0.111 |
| CSE-CIC-IDS-2018 | 0.429 |
| **Overall** | **0.216** |

---

*All benchmarks run on Kaggle GPU (Tesla T4) for multi-dataset results.
Synthetic benchmarks run on standard CPU. Inference latency measured
on a single core with no batching.*
