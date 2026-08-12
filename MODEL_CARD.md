# Model Card: XAI-IDS

## Model Details

- **Model Name**: XAI-IDS (Explainable AI Intrusion Detection System)
- **Model Type**: Multi-class and binary classification
- **Algorithms**: XGBoost, Random Forest, LightGBM, VotingEnsemble, Logistic Regression
- **Version**: 3.0.2
- **Release Date**: April 2026
- **Authors**: Mohammad Thabet Hassan, Fahad Sadek, Ahmed Sami
- **Supervisor**: Dr. Mehak Khurana
- **License**: MIT
- **Repository**: [github.com/MohammadThabetHassan/xai-ids](https://github.com/MohammadThabetHassan/xai-ids)

## Intended Use

### Primary Uses
- **Research**: Studying explainability methods (SHAP, LIME) for intrusion detection
- **Education**: Teaching ML pipeline construction, model evaluation, and XAI techniques
- **Benchmarking**: Comparing model performance across multiple IDS datasets
- **XAI Methodology**: Demonstrating the novel XAI Confidence Score (XCS) for explanation reliability

### Out-of-Scope Uses
- **Production IDS deployment**: This is not a production-ready intrusion detection system
- **Real-time network monitoring**: The pipeline is designed for batch analysis, not real-time inference
- **Standalone security decision-making**: Should be used as part of a defense-in-depth strategy
- **Datasets not evaluated on**: Models trained on CIC-IDS-2017 may not generalize to other datasets (cross-dataset Jaccard = 0.216)

## Datasets

### CIC-IDS-2017
- **Source**: Canadian Institute for Cybersecurity, University of New Brunswick
- **Records**: 2.8+ million flow records
- **Features**: 78 original (20 selected for Kaggle notebook)
- **Classes**: 14 (1 benign + 13 attack types)
- **Attack Types**: DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, DoS slowloris, DoS Slowhttptest, Bot, Web Attack (Brute Force, XSS, Sql Injection), Infiltration
- **Class Imbalance**: Severe (~600:1 ratio between largest and smallest class)

### UNSW-NB15
- **Source**: UNSW Canberra Cyber
- **Records**: 2.5+ million
- **Features**: 20 selected
- **Classes**: 10 (1 normal + 9 attack types)
- **Attack Types**: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms

### CSE-CIC-IDS-2018
- **Source**: Canadian Institute for Cybersecurity
- **Records**: 10+ million
- **Features**: 20 selected
- **Classes**: 2 currently captured (Benign + DDoS attacks-LOIC-HTTP);
  14+ attack types available in the full dataset
- **Type**: Binary (current run); stratified multi-class sampling fix
  applied to notebook for next run

## Model Architecture

### XGBoost (Primary)
- **n_estimators**: 100
- **max_depth**: 8
- **learning_rate**: 0.1
- **tree_method**: hist
- **eval_metric**: mlogloss

### Random Forest
- **n_estimators**: 100
- **max_depth**: 20
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- **class_weight**: balanced

### LightGBM
- Trained on Kaggle with GPU (Tesla T4)
- 20 selected features per dataset

### VotingEnsemble
- Soft voting ensemble of XGBoost, Random Forest, and LightGBM

### Logistic Regression (Main Pipeline)
- **solver**: lbfgs
- **max_iter**: 1000
- **C**: 1.0
- **class_weight**: balanced

## Evaluation Results

### CIC-IDS-2017 (14 classes)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 0.9966 | 0.9964 | 0.9966 | 0.9964 |
| VotingEnsemble | 0.9886 | 0.9945 | 0.9886 | 0.9911 |
| RandomForest | 0.9857 | 0.9940 | 0.9857 | 0.9893 |
| LightGBM | 0.9744 | 0.9930 | 0.9744 | 0.9828 |

### UNSW-NB15 (10 classes)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 0.8004 | 0.8120 | 0.8004 | 0.7982 |
| VotingEnsemble | 0.7867 | 0.8240 | 0.7867 | 0.8002 |
| RandomForest | 0.7635 | 0.8202 | 0.7635 | 0.7848 |
| LightGBM | 0.7630 | 0.8291 | 0.7630 | 0.7863 |

### CSE-CIC-IDS-2018 (Binary)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| RandomForest | 0.9993 | 0.9993 | 0.9993 | 0.9992 |
| VotingEnsemble | 0.9993 | 0.9993 | 0.9993 | 0.9992 |
| XGBoost | 0.9990 | 0.9990 | 0.9990 | 0.9990 |
| LightGBM | 0.9990 | 0.9990 | 0.9990 | 0.9990 |

## XAI Confidence Score (XCS)

The XCS measures explanation reliability:

```
XCS = 0.4 × Confidence + 0.3 × (1 - SHAP_Instability) + 0.3 × Jaccard(SHAP, LIME)
```

- **Threshold**: XCS > 0.3 for acceptable explanation reliability
- **Cross-dataset Jaccard similarity**: 0.216

### XCS Results (v2 — full LIME formula, synthetic test sets, n=100)

| Dataset | Mean XCS | Mean Confidence | Mean SHAP Instability | Mean Jaccard | Flagged (< 0.3) |
|---------|----------|-----------------|----------------------|--------------|-----------------|
| CIC-IDS-2017 | 0.420 | 0.478 | 0.416 | 0.178 | 0 / 100 (0%) |
| UNSW-NB15 | 0.386 | 0.310 | 0.253 | 0.126 | 5 / 100 (5%) |
| CSE-CIC-IDS-2018 | 0.675 | 0.999 | 0.226 | 0.143 | 0 / 100 (0%) |

Previous v2.0.1 CSVs had jaccard_sl=0 for all samples because LIME never ran
during XCS computation. Recomputed with `scripts/recompute_xcs.py`; see
`explanations/xcs_*_v2.csv` for per-sample data.

## Limitations and Biases

1. **Class Imbalance**: Severe imbalance in CIC-IDS-2017 (~600:1 ratio) causes poor detection of minority attack classes even with balanced class weights.

2. **Dataset Specificity**: Cross-dataset feature importance shows low Jaccard similarity (0.216), meaning models trained on one dataset do not generalize well to others.

3. **Feature Selection**: The Kaggle notebook uses 20 features per dataset (selected by importance), while the main pipeline uses all 78 CIC-IDS-2017 features. Results are not directly comparable.

4. **Synthetic Data Default**: The main pipeline defaults to synthetic data. Real data requires download or running the Kaggle notebook.

5. **Temporal Drift**: Network attack patterns evolve over time. Models trained on 2017/2018 data may not detect modern attack variants.

6. **False Positives**: Even with 99.6% accuracy, the extreme class imbalance means false positives can overwhelm alerts in production.

7. **UNSW-NB15 Performance**: 80% accuracy on UNSW-NB15 indicates significant room for improvement, especially on minority attack classes (Analysis, Backdoor, Shellcode, Worms).

8. **CICIDS2018 class diversity**: The current Kaggle notebook run
   captured only 2 of the 14+ available attack classes due to the
   chunked CSV sampler stopping before reaching files with rarer
   attacks. The `load_csv_files_stratified()` function in the updated
   notebook fixes this by sampling proportionally per label value
   within each file.

## Ethical Considerations

- This model is for research and educational purposes only
- Does not constitute a production security solution
- False positives may impact legitimate network traffic analysis
- Should be used as part of a defense-in-depth strategy with human oversight
- Model decisions should not be used for legal or disciplinary actions without human review

## Reproducibility

- Random seed (42) used consistently throughout the pipeline
- Deterministic train/test splitting with stratification
- All preprocessing artifacts (scaler, encoder) saved for inference
- Full pipeline executable via single command: `python run_pipeline.py`
- Multi-dataset results reproducible via Kaggle notebook with GPU

## References

- Sharafaldin, I., Lashkari, A.H. and Ghorbani, A.A., "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," ICISSP 2018.
- Lundberg, S.M. and Lee, S.-I., "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.
- Ribeiro, M.T., Singh, S. and Guestrin, C., "Why Should I Trust You? Explaining the Predictions of Any Classifier," KDD 2016.
- Moustafa, N. and Slay, J., "UNSW-NB15: a comprehensive data set for network intrusion detection systems," MilCIS 2015.
