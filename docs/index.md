# XAI-IDS: Explainable AI Intrusion Detection System

## Overview

XAI-IDS is a multi-dataset Explainable AI-based Intrusion Detection System evaluated on CIC-IDS-2017, UNSW-NB15, and CSE-CIC-IDS-2018. It combines ML classifiers (XGBoost, Random Forest, LightGBM, VotingEnsemble) with SHAP and LIME explainability, featuring a novel XAI Confidence Score (XCS).

## Key Features

- **Multi-dataset**: CIC-IDS-2017 (14 classes), UNSW-NB15 (10 classes), CSE-CIC-IDS-2018 (binary)
- **Multi-model**: XGBoost, Random Forest, LightGBM, VotingEnsemble
- **XCS Metric**: Novel XAI Confidence Score measuring explanation reliability
- **SHAP**: Global feature importance with beeswarm, waterfall, summary plots
- **LIME**: Local instance-level explanations
- **Jaccard Similarity**: SHAP vs LIME agreement analysis
- **Production-ready**: Docker, FastAPI inference endpoint

## Best Results

| Dataset | Model | Accuracy |
|---------|-------|----------|
| CIC-IDS-2017 | XGBoost | 99.66% |
| UNSW-NB15 | XGBoost | 80.04% |
| CSE-CIC-IDS-2018 | RandomForest | 99.93% |

## Getting Started

```bash
git clone https://github.com/MohammadThabetHassan/xai-ids.git
cd xai-ids
pip install -r requirements.txt
python run_pipeline.py
```

## Documentation

- [Methodology](methodology.md) - XCS formula and technical details
- [Full Results](../RESULTS.md) - Complete metrics tables
- [Model Card](../MODEL_CARD.md) - Model details and limitations
- [GitHub Pages](https://mohammadthabethassan.github.io/xai-ids/) - Interactive demo

## Links

- **Repository**: [GitHub](https://github.com/MohammadThabetHassan/xai-ids)
- **Datasets**: [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) | [UNSW-NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)