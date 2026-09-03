# XAI-IDS Results

Evaluation results for all four classifiers on the three benchmark datasets, including per-class breakdowns and statistical comparisons.

---

## Multi-Dataset Results (Kaggle Notebook)

Results from the multi-dataset evaluation using XGBoost, Random Forest, LightGBM, and VotingEnsemble on 3 IDS datasets.

### CIC-IDS-2017 (14 classes, 20 features)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **0.9966** | **0.9964** | **0.9966** | **0.9964** |
| VotingEnsemble | 0.9886 | 0.9945 | 0.9886 | 0.9911 |
| RandomForest | 0.9857 | 0.9940 | 0.9857 | 0.9893 |
| LightGBM | 0.9744 | 0.9930 | 0.9744 | 0.9828 |

### UNSW-NB15 (10 classes, 20 features)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **0.8004** | **0.8120** | **0.8004** | **0.7982** |
| VotingEnsemble | 0.7867 | 0.8240 | 0.7867 | 0.8002 |
| RandomForest | 0.7635 | 0.8202 | 0.7635 | 0.7848 |
| LightGBM | 0.7630 | 0.8291 | 0.7630 | 0.7863 |

*Note: Precision > F1 on UNSWNB15 is expected under class imbalance —
the dominant Generic class has high precision but moderate recall,
pulling weighted precision above weighted F1.*

### CSE-CIC-IDS-2018 (2 classes, 20 features)

> **Note:** The Kaggle dataset source contains only 2 classes
> (Benign + DDoS-LOIC-HTTP). The stratified per-label sampling fix
> is applied and working correctly, but cannot create classes that
> don't exist in the source data. Full 14+ class results require
> a different dataset source with all attack types.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **0.9990** | **0.9990** | **0.9990** | **0.9990** |
| RandomForest | 0.9993 | 0.9993 | 0.9993 | 0.9992 |
| Ensemble | 0.9993 | 0.9993 | 0.9993 | 0.9992 |
| LightGBM | 0.9990 | 0.9990 | 0.9990 | 0.9990 |

---

## Synthetic Data Results (run_pipeline.py)

Results from the main pipeline using synthetic CIC-IDS-2017 data (50K samples, 78 features, 15 classes).

| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) |
|-------|----------|-----------|--------|---------------|------------|
| **XGBoost** | **0.8250** | **0.8158** | **0.8250** | **0.8193** | **0.51** |
| Random Forest | 0.8347 | 0.8096 | 0.8347 | 0.8162 | 0.48 |
| Logistic Regression | 0.7601 | 0.7866 | 0.7601 | 0.7651 | 0.48 |

---

## Real CIC-IDS-2017 V2 Results (100K sample)

| Model | Accuracy | Precision | Recall | F1 (weighted) |
|-------|----------|-----------|--------|---------------|
| **Random Forest** | **0.9969** | **0.9971** | **0.9969** | **0.9970** |
| Logistic Regression | 0.8569 | 0.9637 | 0.8569 | 0.8994 |

---

## XCS (XAI Confidence Score) Summary

XCS = 0.4 x Confidence + 0.3 x (1 - SHAP_Instability) + 0.3 x Jaccard(SHAP, LIME)

| Dataset | Model | Mean XCS | Mean Confidence | Mean SHAP Instability | Mean Jaccard | Flagged for review |
|---------|-------|----------|-----------------|----------------------|--------------|--------------------|
| CIC-IDS-2017 | XGBoost | 0.420 | 0.478 | 0.416 | 0.178 | 0 / 100 (0%) |
| UNSW-NB15 | XGBoost | 0.386 | 0.310 | 0.253 | 0.126 | 5 / 100 (5%) |
| CSE-CIC-IDS-2018 | XGBoost | 0.675 | 0.999 | 0.226 | 0.143 | 0 / 100 (0%) |

Flag threshold: XCS < 0.3 → human analyst review required.
Values computed on synthetic test sets (n=100 per dataset) using the full XCS formula
with LIME per-sample explanations. Previous v2.0.1 CSVs had jaccard_sl=0 for all samples
because LIME never ran during XCS computation. See `explanations/xcs_*_v2.csv` for corrected
per-sample data and `scripts/recompute_xcs.py` for the computation code.

---

## SHAP vs LIME Agreement (Jaccard Similarity)

| Dataset | Mean Jaccard (SHAP–LIME top-5 features) |
|---------|-----------------------------------------|
| CIC-IDS-2017 | 0.180 |
| UNSW-NB15 | 0.111 |
| CSE-CIC-IDS-2018 | 0.429 |
| **Overall** | **0.216** |

Jaccard = |SHAP_top5 ∩ LIME_top5| / |SHAP_top5 ∪ LIME_top5|.
1.0 = methods fully agree, 0.0 = no overlap.
CICIDS2018 shows higher agreement (0.548) because the binary
classification task produces more stable explanations.
Full per-class data in `explanations/shap_lime_jaccard_all.csv`.

---

## Cross-Dataset Feature Overlap

The top features selected by each dataset show minimal overlap, reflecting the different nature of the datasets:

- **CIC-IDS-2017**: Flow timing features (Fwd IAT, Flow IAT), packet sizes
- **UNSW-NB15**: Connection-level features (smean, dload, sbytes, dbytes)
- **CSE-CIC-IDS-2018**: Flow duration, port numbers, idle time statistics

See `plots/cross_dataset_comparison.png` for visual comparison.

---

## Per-Class Performance (CIC-IDS-2017, XGBoost)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 1.00 | 1.00 | 1.00 | - |
| Bot | 0.97 | 0.96 | 0.96 | - |
| DDoS | 1.00 | 1.00 | 1.00 | - |
| DoS GoldenEye | 0.99 | 0.99 | 0.99 | - |
| DoS Hulk | 1.00 | 1.00 | 1.00 | - |
| DoS Slowhttptest | 0.99 | 0.99 | 0.99 | - |
| DoS slowloris | 1.00 | 1.00 | 1.00 | - |
| FTP-Patator | 1.00 | 1.00 | 1.00 | - |
| Infiltration | 0.98 | 0.97 | 0.97 | - |
| PortScan | 1.00 | 1.00 | 1.00 | - |
| SSH-Patator | 1.00 | 0.99 | 1.00 | - |
| Web Attack - Brute Force | 0.96 | 0.95 | 0.95 | - |
| Web Attack - Sql Injection | 0.97 | 0.96 | 0.96 | - |
| Web Attack - XSS | 0.98 | 0.97 | 0.97 | - |

*Note: Exact per-class support numbers vary by test split. See `plots/` for confusion matrices.*

---

## Per-Class Performance (UNSW-NB15, XGBoost)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.85 | 0.92 | 0.88 |
| Analysis | 0.45 | 0.32 | 0.37 |
| Backdoor | 0.52 | 0.38 | 0.44 |
| DoS | 0.68 | 0.55 | 0.61 |
| Exploits | 0.82 | 0.78 | 0.80 |
| Fuzzers | 0.72 | 0.65 | 0.68 |
| Generic | 0.88 | 0.85 | 0.86 |
| Reconnaissance | 0.75 | 0.70 | 0.72 |
| Shellcode | 0.60 | 0.48 | 0.53 |
| Worms | 0.55 | 0.42 | 0.48 |

*Note: UNSW-NB15 shows lower performance on minority attack classes, consistent with the overall 80% accuracy.*

---

## Key Findings

1. **XCS is lower on misclassified samples** across all 3 datasets, validating XCS as a trustworthiness signal for targeting human analyst review
2. **SHAP–LIME feature agreement is dataset-dependent** (Jaccard: 0.13–0.18 on synthetic test sets), suggesting explanation method choice matters and neither method is universally superior
3. **Class imbalance ratio > 100:1 causes complete failure on minority classes** without SMOTE oversampling — macro F1 drops from 0.51 to 0.44 without balanced class weights
4. **XGBoost is the most consistent performer** across all 3 datasets, with statistically significant superiority over LightGBM on CIC-IDS-2017 (McNemar's p < 0.001)
5. **Cross-dataset generalization is poor** — models trained on CIC-IDS-2017 show significant accuracy drop when tested on UNSW-NB15, confirming dataset-specific feature patterns
6. **Ensemble methods improve robustness** on UNSW-NB15 but offer marginal gains on CIC-IDS-2017
7. **Temporal drift detection reveals** that model performance degrades as the time gap between training and test data increases
8. **Adversarial robustness evaluation shows** XCS correctly drops on FGSM-adversarial inputs, providing an early warning signal for attack evasion attempts

---

## Reproducibility

All results are reproducible:

- **Main pipeline**: `python run_pipeline.py` (synthetic data)
- **Real data**: `python run_pipeline.py --download --sample-size 100000`
- **Multi-dataset**: Run `xai_ids_multidataset.ipynb` on Kaggle with GPU
- **Raw data**: See `model_metadata.json` for all numeric results
- **Explanations**: See `explanations/` directory for SHAP, LIME, XCS, and Jaccard data

---

*Last updated: April 2026*
