# Versioned demo bundle

Each tagged release publishes a lightweight source bundle with a manifest containing the release tag, source commit, entry points, and excluded paths.

## Entry points

Use [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the supported synthetic-data quick start. The main commands are:

```bash
python -m pytest tests/test_smoke.py -v -x
python run_pipeline.py --sample-size 5000
```

The API demo entry point is [`api/app.py`](../api/app.py). Start it from the repository root with:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The release bundle also includes [`examples/predict_example.py`](../examples/predict_example.py), the model card, full results, source code, tests, and documentation.

## Bundle boundaries

The release artifact does not include raw datasets, generated outputs, trained model weights, virtual environments, or dependency caches. Obtain those components through their documented sources and review the applicable dataset, model, and third-party licenses before redistribution.
