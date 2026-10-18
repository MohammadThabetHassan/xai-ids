# Re-run plan

## 1. Kaggle setup
1. Create a new Kaggle notebook (Python). In the right-hand **Settings** panel, turn **Internet on**. GPU is optional: training uses it, but SHAP, LIME and the latency timings run on CPU.
2. Attach these datasets with **Add data**:
   - `dhoogla/cicids2017` (parquet, cleaned version of CIC-IDS-2017)
   - `dhoogla/unswnb15` (parquet, `attack_cat` label)
   - `solarmainframe/ids-intrusion-csv` (**all ten** CSE-CIC-IDS-2018 day files). Do not use the old `mohdqwe123/dataset1/csecicids2018` folder.

   These names were taken from the original notebook and have not been re-checked on Kaggle. If one can't be found, search Kaggle for the dataset (CIC-IDS-2017, UNSW-NB15 with `attack_cat`, or CSE-CIC-IDS-2018 with all ten days) and note which one you used. If the script prints `NOT FOUND` for a dataset, run `!ls /kaggle/input` to see where it was mounted. Then add that path to the matching list in `PATHS` near the top of the script.
3. In the first cell, install the libraries that may be missing from the Kaggle image:
   ```
   !pip install -q lime imbalanced-learn
   ```
4. Paste `revision/xai_ids_revised.py` into the second cell.
5. Run it with **Save Version → Save & Run All (Commit)**, not interactively. The run takes about 1.5–2.5 h, mostly LIME, and an interactive session can stop when idle. A saved version runs in the background and keeps its output. If it runs out of time, set `LIME_SAMPLES = 500` and note that in the paper.

## 2. Check the log before trusting any number
- `Loading CICIDS2018` lists all ten files, and **the class list has more than 2 classes** (expect about 15, e.g. Bot, FTP/SSH brute force, DoS GoldenEye/Hulk/Slowloris/SlowHTTPTest, DDoS LOIC-HTTP/LOIC-UDP/HOIC, Infilteration, web attacks).
- `columns not shared by all files (dropped)` lists only ID-type columns.
- The `cleaning log` for each dataset gives raw → final rows. These go into the paper's dataset table.
- No feature list contains `Port`.

## 3. What to bring back
1. When the saved version finishes, open it. From its **Output** tab, download the `xai_ids_revised/` folder. From its **Logs** tab, download the full log.
2. Hand the results back:
   - **With write access to this repo:** commit the folder's contents, except `models/` (too large), under `results/revised/` on a new branch, for example `results/revised-run`. Save the log as `results/revised/run_log.txt`. Open a pull request. In the description, give the Kaggle version number and say whether `LIME_SAMPLES`, a dataset or any other setting was changed.
   - **Without write access:** zip the folder (without `models/`) together with the log, and send it to the paper's lead with the same notes.

The paper will be rewritten from these files only:

| File | Used for |
|---|---|
| `summary.json` | Dataset sizes, splits, SMOTE, classes, features, XCS thresholds |
| `tables/model_comparison.csv` | Main results table (accuracy, weighted/macro F1 with CIs, balanced acc., ECE) |
| `tables/per_class_*_XGBoost.csv` | Per-class tables |
| `tables/mcnemar.csv` | Significance statements |
| `tables/xcs_evaluation.csv` | **The core XCS result**: does XCS beat confidence? |
| `tables/shap_lime_agreement.csv`, `shap_family_share.csv`, `cross_dataset_feature_overlap.csv` | Explainability section |
| `tables/xcs_latency.csv` | Any latency claim |
| `plots/*.png` | Every figure in the paper (replaces the current Figs. 2, 3, 4, 8, 11, 12) |

## 4. How the result decides the paper's framing
Look at `xcs_evaluation.csv`, rows `xcs_fixed` and `xcs_learned` vs `confidence`:
- **The `auroc_minus_conf` CI excludes 0 (positive)**: XCS adds information beyond confidence. The XCS contribution stands, now properly validated.
- **The CI includes 0, or the difference is negative**: XCS does not improve on confidence for error detection. The paper becomes an honest negative/analysis result ("explanation stability and SHAP–LIME agreement do not add error-detection signal beyond confidence on these benchmarks"). That is still publishable and much safer than the current claims. The smoke test already hints this could happen, but its data is synthetic, so this is unknown until the real run.

## 5. Optional, if time allows
- Repeat with `RANDOM_STATE` in {7, 2024} and report mean ± sd.
- Run the adversarial module (`src/evaluation/adversarial.py`) and keep it in the paper only if real results exist. Otherwise remove it from the paper.
