# Results of the corrected pipeline

All runs below come from the Kaggle notebook `omaralraas/xai-ids-corrected-rerun` (4 vCPU / 31 GB RAM, no accelerator).

| Folder | Run | What it is |
|---|---|---|
| `revised/` | Version 3, commit `5a8053e`, seed 42, `XAI_IDS_SENS=1`, 77.5 min | **Main results.** Imbalance handling chosen per dataset on validation data (CHANGES.md #23), entropy and trust-score baselines (#24), XCS sensitivity analysis (#26). |
| `revised/paper/` | `revision/analyze_results.py results/revised results/revised/paper --seeds results/seeds/seed7 results/seeds/seed2024 --temporal results/temporal2018` | Every table (CSV) and figure (PNG) used in the paper, including the seed, temporal and sensitivity tables. |
| `seeds/seed7/`, `seeds/seed2024/` | Version 3, `XAI_IDS_SEED=7` / `2024`, 62 / 64 min | Full repeats with a different seed for sampling, splitting, feature selection, training and the XCS samples (#25). |
| `temporal2018/` | Version 3, `XAI_IDS_SPLIT=temporal`, seed 42, 23 min | CSE-CIC-IDS-2018 only, split chronologically within each (capture file, label): earliest 64% train, next 16% validation, latest 20% test (#25). |
| `kaggle_v3_status.json` | Version 3 | Return code, runtime and settings of each of the four runs. |
| `run1_imbalance-both/` | Version 1, commit `6c16c09`, 62 min | First corrected run, with SMOTE and class weights both always applied. Kept because the paper discloses it (Section 5.1 and Threats to Validity). Its plots are omitted. |

Version 2 (commit `4a56e29`, results commit `1f19c2f`) was the main run before the baselines were added. With seed 42, version 3 trains exactly the same models: all twelve model–dataset accuracy and macro F1 values are identical. The XCS numbers differ slightly (by at most 0.026 AUROC) because the trust-score step draws random numbers, which changes the XCS evaluation samples. Version 2's files remain in the git history at `1f19c2f`.

Each run folder contains:

- `summary.json`: configuration, cleaning log, splits, classes, features, imbalance mode and XCS thresholds.
- `tables/`: model comparison, per-class metrics, confusion matrices, McNemar tests, SHAP importance, SHAP–LIME agreement, XCS evaluation (including the baselines) and latency. `revised/tables/xcs_sensitivity.csv` holds the sensitivity analysis.
- `xcs/`: per-sample XCS components and baseline scores for the validation and test samples.
- `plots/`: the pipeline's own figures.
- `run_log.txt`: the Kaggle log of that run.

Trained models are not committed because of their size; rerun the pipeline to regenerate them.

## Headline numbers (`revised/`, seed 42)

| Dataset | Classes | Test rows | Best macro F1 [95% CI] | Confidence AUROC for errors | Trust score Δ AUROC | XCS (a priori) Δ AUROC | XCS (learned) Δ AUROC |
|---|---|---|---|---|---|---|---|
| CIC-IDS-2017 | 15 | 40,000 | 0.836 [0.810, 0.895] (Ensemble) | 0.982 | −0.222 [−0.250, −0.192] | −0.115 [−0.133, −0.097] | −0.022 [−0.034, −0.014] |
| UNSW-NB15 | 10 | 14,078 | 0.697 [0.674, 0.714] (Ensemble) | 0.852 | −0.090 [−0.109, −0.071] | −0.043 [−0.055, −0.032] | −0.0021 [−0.0051, 0.0005] |
| CSE-CIC-IDS-2018 | 15 | 26,301 | 0.813 [0.785, 0.834] (LightGBM) | 0.861 | −0.105 [−0.124, −0.085] | −0.059 [−0.075, −0.044] | −0.0038 [−0.0071, −0.0005] |

Δ AUROC is the paired bootstrap difference from model confidence. XCS is never better than confidence at detecting the model's own errors.

## Robustness

- **Seeds** (`revised/paper/table_seeds.csv`, `table_seeds_runs.csv`): over 3 seeds × 3 datasets, the a-priori XCS is significantly worse than confidence in 9/9 cases and the learned XCS is never significantly better. Mean confidence AUROC is 0.980 ± 0.002, 0.857 ± 0.007 and 0.860 ± 0.001.
- **Chronological split** (`table_temporal*.csv`): on CSE-CIC-IDS-2018, best macro F1 0.795 (XGBoost), confidence AUROC 0.853, a-priori XCS Δ −0.102 [−0.122, −0.084].
- **XCS settings** (`table_xcs_sensitivity.csv`): varying σ ∈ {0.01, 0.05, 0.2}, top-k ∈ {3, 5, 10} and LIME samples ∈ {500, 1000, 2000} one at a time, the a-priori XCS is significantly worse than confidence in 27/27 cases.
