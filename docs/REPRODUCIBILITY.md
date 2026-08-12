# Reproducibility Guide

This guide defines the smallest local path for validating the xai-ids pipeline without downloading the public IDS datasets. The default path uses the repository’s synthetic sample generator and is intended to verify imports, preprocessing, training, evaluation, and output creation.

## Environment

Use Python 3.10 or newer in an isolated virtual environment. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,api]"
```

The editable install uses the dependency declarations in `pyproject.toml`. Keep the generated `.venv/` directory outside version control.

## Quick start

Run the smoke suite first:

```bash
python -m pytest tests/test_smoke.py -v -x
```

Then run the small synthetic pipeline:

```bash
python run_pipeline.py --sample-size 5000
```

The equivalent Make target is:

```bash
make pipeline-small
```

The smoke suite validates module imports, synthetic data generation, preprocessing, metrics, and a mini end-to-end execution. The small pipeline exercises the full training and evaluation path with a bounded synthetic sample.

## Expected results

A successful run should satisfy all of the following conditions:

| Check | Expected result |
|---|---|
| Smoke tests | The command exits with status 0 and reports no failed tests. |
| Pipeline exit status | `python run_pipeline.py --sample-size 5000` exits with status 0. |
| Metrics artifact | `outputs/results_metrics.csv` exists and contains model evaluation metrics. |
| Evaluation outputs | The configured output directories contain generated reports and figures when the corresponding pipeline stages run. |
| No dataset download | The quick-start path does not require `--download` or files under `data/raw/`. |

Verify the primary artifact with:

```bash
test -s outputs/results_metrics.csv
head -n 5 outputs/results_metrics.csv
```

Exact metric values can change with dependency versions, random seeds, and implementation changes. Treat the exit status, artifact schema, and test results as the reproducibility gate; compare numerical results against [`RESULTS.md`](../RESULTS.md) only when the same dataset, feature set, seed, and pipeline options are used.

## Full tests and real datasets

Run the complete test suite with:

```bash
python -m pytest tests/ -v --tb=short
```

The real-data path requires the dataset download option and the associated dataset terms and access conditions:

```bash
python run_pipeline.py --download
```

Results from the CIC-IDS-2017, UNSW-NB15, and CSE-CIC-IDS-2018 paths are not directly interchangeable. The main pipeline and the notebooks may use different feature sets, preprocessing choices, and dataset splits. Record the dataset, feature count, split, seed, and command line in any benchmark report.

## Troubleshooting

If the smoke suite fails during import, confirm that the editable install completed in the active virtual environment. If the pipeline fails because an output path is stale, remove generated artifacts with `make clean` and rerun the quick start. Do not commit generated models, raw datasets, local logs, or environment files unless a change explicitly requires a versioned artifact.
