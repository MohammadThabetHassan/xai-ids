# Results of the corrected pipeline

| Folder | Run | What it is |
|---|---|---|
| `revised/` | Kaggle notebook `omaralraas/xai-ids-corrected-rerun`, version 2, commit `4a56e29`, 69.5 min, 4 vCPU / 31 GB RAM, no accelerator | **Main results.** Imbalance handling chosen per dataset on validation data (CHANGES.md #23). |
| `revised/paper/` | `revision/analyze_results.py results/revised results/revised/paper` | Every table (CSV) and figure (PNG) used in the revised paper. |
| `run1_imbalance-both/` | Same notebook, version 1, commit `6c16c09`, 62 min | First corrected run, with SMOTE and class weights both always applied. Kept because the paper discloses it (Section 5.1 and Threats to Validity). Its plots are omitted. |

Each run folder contains:

- `summary.json`: configuration, cleaning log, splits, classes, features, imbalance mode and XCS thresholds.
- `tables/`: model comparison, per-class metrics, confusion matrices, McNemar tests, SHAP importance, SHAP–LIME agreement, XCS evaluation and latency.
- `xcs/`: per-sample XCS components for the validation and test samples.

`revised/` also has `plots/` (the pipeline's own figures) and `run_log.txt` (the Kaggle log). Trained models are not committed because of their size; rerun the pipeline to regenerate them.

## Headline numbers (`revised/`)

| Dataset | Classes | Test rows | Best macro F1 [95% CI] | Confidence AUROC for errors | XCS (a priori) Δ AUROC | XCS (learned) Δ AUROC |
|---|---|---|---|---|---|---|
| CIC-IDS-2017 | 15 | 40,000 | 0.836 [0.810, 0.895] (Ensemble) | 0.981 | −0.127 [−0.147, −0.108] | −0.027 [−0.037, −0.018] |
| UNSW-NB15 | 10 | 14,078 | 0.697 [0.674, 0.714] (Ensemble) | 0.849 | −0.032 [−0.045, −0.021] | −0.000 [−0.002, 0.001] |
| CSE-CIC-IDS-2018 | 15 | 26,301 | 0.813 [0.785, 0.834] (LightGBM) | 0.867 | −0.049 [−0.063, −0.037] | −0.005 [−0.009, −0.001] |

Δ AUROC is the paired bootstrap difference from model confidence. XCS is never better than confidence at detecting the model's own errors. See `revised/paper/` and the revised paper for details.
