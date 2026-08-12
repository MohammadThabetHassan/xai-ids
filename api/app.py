"""
FastAPI inference endpoint for XAI-IDS.

Provides REST API for real-time intrusion detection predictions
with SHAP explanations and full XCS (XAI Confidence Score) computation.

XCS Formula (real-time):
    XCS = 0.4*Confidence + 0.3*(1-SHAP_Instability) + 0.3*Jaccard(SHAP,LIME)

At inference time:
- Confidence: model softmax probability for predicted class
- SHAP Instability: two SHAP calls with different random background
  subsamples (n=50 each), Spearman rank correlation of top-10 features,
  instability = 1 - correlation
- LIME-SHAP Jaccard: LIME with num_samples=300, top-5 features from
  both methods, Jaccard similarity of feature name sets

Latency tradeoff: Full XCS adds ~200-500ms per request due to
dual SHAP + LIME computation. Use /predict for fast-path (confidence
only, ~10ms) and /explain for full XCS with explanations.
"""

import json
import math
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger("xai_ids.api")

# Support both pipeline outputs/models and Kaggle models/ directory
MODELS_DIR = Path("models")
PIPELINE_MODELS_DIR = Path("outputs/models")

models = {}
scaler = None
label_encoder = None
feature_names = None
training_stats = None  # Stored feature bounds for drift detection
xcs_weights = None      # Learned XCS weights from pipeline


def _load_learned_xcs_weights():
    """Load learned XCS weights if available."""
    global xcs_weights
    for weights_path in [
        Path("outputs/reports/learned_xcs_weights.json"),
        Path("reports/learned_xcs_weights.json"),
    ]:
        if weights_path.exists():
            try:
                with open(weights_path) as f:
                    data = json.load(f)
                xcs_weights = data.get("normalized_weights", [0.4, 0.3, 0.3])
                logger.info(f"Loaded learned XCS weights: {xcs_weights}")
                return
            except Exception as e:
                logger.warning(f"Failed to load learned XCS weights: {e}")
    xcs_weights = [0.4, 0.3, 0.3]
    logger.info("Using default XCS weights: [0.4, 0.3, 0.3]")


class PredictionInput(BaseModel):
    features: List[float] = Field(
        ...,
        description="Network flow features (78 for CIC-IDS-2017, 20 for Kaggle models)",
        min_length=1,
        max_length=100,
    )

    @field_validator("features")
    @classmethod
    def reject_non_finite_features(cls, values: List[float]) -> List[float]:
        if not all(math.isfinite(value) for value in values):
            raise ValueError("features must contain only finite numbers")
        return values


def _validate_feature_count(features: np.ndarray) -> None:
    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and features.shape[1] != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: expected {expected} features, got {features.shape[1]}",
        )


class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    xcs_score: Optional[float] = Field(
        None,
        description=(
            "XAI Confidence Score (fast-path: confidence component only = 0.4×Conf). "
            "Full XCS = 0.4×Conf + 0.3×(1-SHAP_Instability) + 0.3×Jaccard(SHAP,LIME) "
            "is computed in /explain endpoint. XCS < 0.3 flags for human review."
        )
    )


class ExplanationOutput(BaseModel):
    prediction: str
    confidence: float
    top_features: List[dict]
    xcs_score: Optional[float] = None
    xcs_components: Optional[dict] = None
    xcs_reliable: Optional[bool] = None


def _compute_shap_instability(model, X: np.ndarray, n_background: int = 50, random_state: int = 42) -> float:
    """Compute SHAP instability by comparing two SHAP runs with different background samples.

    Parameters
    ----------
    model : trained tree model
    X : np.ndarray
        Single sample, shape (1, n_features).
    n_background : int
        Number of background samples for each SHAP run.
    random_state : int
        Random seed.

    Returns
    -------
    float
        Instability = 1 - Spearman correlation of top-10 feature ranks.
        0 = perfectly stable, 1 = completely unstable.
    """
    from scipy import stats

    rng = np.random.RandomState(random_state)

    # Generate synthetic background data for SHAP
    n_features = X.shape[1]
    bg_a = rng.randn(n_background, n_features).astype(np.float32)
    bg_b = rng.randn(n_background, n_features).astype(np.float32)

    try:
        import shap
        explainer_a = shap.TreeExplainer(model, data=bg_a)
        explainer_b = shap.TreeExplainer(model, data=bg_b)

        sv_a = explainer_a.shap_values(X)
        sv_b = explainer_b.shap_values(X)

        # Handle multi-class
        if isinstance(sv_a, list):
            pred = model.predict(X)[0]
            sv_a = sv_a[pred][0]
            sv_b = sv_b[pred][0]
        elif sv_a.ndim == 3:
            pred = model.predict(X)[0]
            sv_a = sv_a[0, :, pred]
            sv_b = sv_b[0, :, pred]
        else:
            sv_a = sv_a[0]
            sv_b = sv_b[0]

        # Rank top-10 features by absolute SHAP value
        top_k = min(10, len(sv_a))
        rank_a = np.argsort(np.abs(sv_a))[-top_k:]
        rank_b = np.argsort(np.abs(sv_b))[-top_k:]

        # Spearman correlation of ranks
        if len(rank_a) < 3:
            return 0.5

        corr, _ = stats.spearmanr(rank_a, rank_b)
        if np.isnan(corr):
            return 0.5

        instability = 1.0 - max(0.0, corr)
        return float(instability)

    except Exception:
        return 0.5


def _compute_lime_shap_jaccard(
    model, X: np.ndarray, feature_names: List[str], n_lime_samples: int = 300, n_top: int = 5
) -> float:
    """Compute Jaccard similarity between LIME and SHAP top-n features.

    Parameters
    ----------
    model : trained classifier
    X : np.ndarray
        Single sample, shape (1, n_features).
    feature_names : list of str
    n_lime_samples : int
        Number of LIME perturbation samples.
    n_top : int
        Number of top features to compare.

    Returns
    -------
    float
        Jaccard similarity between SHAP top-n and LIME top-n feature names.
    """
    try:
        import lime.lime_tabular
        import shap

        # SHAP top-n
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            pred = model.predict(X)[0]
            sv = sv[pred][0]
        elif sv.ndim == 3:
            pred = model.predict(X)[0]
            sv = sv[0, :, pred]
        else:
            sv = sv[0]

        shap_top_idx = np.argsort(np.abs(sv))[-n_top:]
        shap_top_names = {feature_names[i] for i in shap_top_idx}

        # LIME top-n
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names,
            mode="classification",
        )
        exp = lime_explainer.explain_instance(
            X[0],
            model.predict_proba,
            num_features=n_top,
            num_samples=n_lime_samples,
        )

        as_map = exp.as_map()
        name_to_weight = {}
        for cls_id in as_map:
            for feat_idx, weight in as_map[cls_id]:
                feat_name = feature_names[int(feat_idx)]
                name_to_weight[feat_name] = name_to_weight.get(feat_name, 0.0) + abs(weight)

        sorted_feats = sorted(name_to_weight, key=lambda k: name_to_weight[k], reverse=True)
        lime_top_names = set(sorted_feats[:n_top])

        # Jaccard
        if not shap_top_names and not lime_top_names:
            return 0.0
        return len(shap_top_names & lime_top_names) / len(shap_top_names | lime_top_names)

    except Exception:
        return 0.0


def compute_realtime_xcs(
    model,
    X: np.ndarray,
    feature_names: List[str],
    confidence: float,
) -> dict:
    """Compute full XCS score in real-time.

    Parameters
    ----------
    model : trained classifier
    X : np.ndarray
        Single sample, shape (1, n_features).
    feature_names : list of str
    confidence : float
        Model confidence (max predict_proba).

    Returns
    -------
    dict
        Contains xcs_score, components (confidence, instability, jaccard),
        and xcs_reliable flag.
    """
    start = time.time()

    instability = _compute_shap_instability(model, X)
    jaccard = _compute_lime_shap_jaccard(model, X, feature_names)

    xcs = 0.4 * confidence + 0.3 * (1 - instability) + 0.3 * jaccard

    elapsed = time.time() - start
    logger.info(
        f"Real-time XCS: conf={confidence:.4f}, instab={instability:.4f}, "
        f"jac={jaccard:.4f}, xcs={xcs:.4f} ({elapsed:.2f}s)"
    )

    return {
        "xcs_score": round(float(xcs), 4),
        "components": {
            "confidence": round(float(confidence), 4),
            "shap_instability": round(float(instability), 4),
            "jaccard_sl": round(float(jaccard), 4),
        },
        "xcs_reliable": bool(xcs > 0.7),
        "computation_time_ms": round(elapsed * 1000, 1),
    }


def load_models():
    """Load trained models and preprocessors from available directories."""
    global models, scaler, label_encoder, feature_names, training_stats

    # Try Kaggle models/ directory first, then pipeline outputs/models/
    model_dirs = [MODELS_DIR, PIPELINE_MODELS_DIR]

    for model_dir in model_dirs:
        if not model_dir.exists():
            continue

        # Load models (try various naming conventions)
        model_patterns = {
            "xgboost": [
                "xgb_CICIDS2017.joblib", "xgboost_CICIDS2017.joblib",
                "xgboost.pkl",
            ],
            "random_forest": [
                "rf_CICIDS2017.joblib", "rf_UNSWNB15.joblib",
                "rf_CICIDS2018.joblib", "random_forest.pkl",
            ],
            "lightgbm": [
                "lgb_CICIDS2017.joblib", "lightgbm_CICIDS2017.joblib",
                "lightgbm_UNSWNB15.joblib", "lightgbm_CICIDS2018.joblib",
            ],
            "logistic_regression": ["logistic_regression.pkl"],
        }

        for name, patterns in model_patterns.items():
            for pattern in patterns:
                path = model_dir / pattern
                if path.exists() and name not in models:
                    try:
                        models[name] = joblib.load(path)
                        logger.info(f"Loaded {name} from {path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load {path}: {e}")

        # Load scaler
        for scaler_name in ["scaler_CICIDS2017.joblib", "scaler.pkl"]:
            scaler_path = model_dir / scaler_name
            if scaler_path.exists() and scaler is None:
                try:
                    scaler = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler from {scaler_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load scaler {scaler_path}: {e}")

        # Load label encoder
        for le_name in ["label_encoder_CICIDS2017.joblib", "label_encoder.pkl"]:
            le_path = model_dir / le_name
            if le_path.exists() and label_encoder is None:
                try:
                    label_encoder = joblib.load(le_path)
                    logger.info(f"Loaded label encoder from {le_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load label encoder {le_path}: {e}")

        # Load feature names
        for feat_name in ["features_CICIDS2017.joblib"]:
            feat_path = model_dir / feat_name
            if feat_path.exists() and feature_names is None:
                try:
                    feature_names = joblib.load(feat_path)
                    logger.info(f"Loaded feature names from {feat_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load features {feat_path}: {e}")

    if models:
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    else:
        logger.warning("No models found in models/ or outputs/models/")

    if scaler is not None:
        logger.info("Scaler loaded")
    if label_encoder is not None:
        logger.info(f"Label encoder loaded with {len(label_encoder.classes_)} classes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown."""
    load_models()
    yield
    # Cleanup on shutdown
    models.clear()


app = FastAPI(
    title="XAI-IDS API",
    description="Explainable AI Intrusion Detection System API",
    version="3.0.2",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "XAI-IDS API",
        "version": "3.0.2",
        "models_loaded": list(models.keys()),
        "scaler_loaded": scaler is not None,
        "label_encoder_loaded": label_encoder is not None,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models) > 0,
        "models": list(models.keys()),
        "scaler_loaded": scaler is not None,
        "encoder_loaded": label_encoder is not None,
    }


@app.get("/health/features")
async def health_features():
    """Check feature distribution bounds for drift detection.

    Returns stored training feature statistics (mean, std, min, max)
    so clients can check for feature drift before sending predictions.
    """
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")

    result = {
        "scaler_type": type(scaler).__name__,
        "n_features": int(scaler.n_features_in_),
        "feature_means": [round(float(m), 6) for m in scaler.mean_],
        "feature_stds": [round(float(s), 6) for s in scaler.scale_],
    }

    if feature_names:
        result["feature_names"] = feature_names

    return result


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make a prediction for a single network flow.

    Parameters
    ----------
    input_data : PredictionInput
        Network flow features.

    Returns
    -------
    PredictionOutput
        Prediction with confidence, probabilities, and XCS fast-path score.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")

    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    try:
        features = np.array(input_data.features, dtype=np.float64).reshape(1, -1)
        _validate_feature_count(features)
        features_scaled = scaler.transform(features)

        model = models.get("random_forest") or models.get("xgboost") or models.get("lightgbm")
        if not model:
            raise HTTPException(status_code=503, detail="No prediction model loaded")

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])

        prob_dict = {
            str(label_encoder.inverse_transform([i])[0]): float(p)
            for i, p in enumerate(probabilities)
        }

        # Fast-path XCS: confidence component only
        xcs_score = round(0.4 * confidence, 4)

        return PredictionOutput(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            xcs_score=xcs_score,
        )

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: expected {scaler.n_features_in_} features, got {len(input_data.features)}"
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed") from None


@app.post("/explain", response_model=ExplanationOutput)
async def explain(input_data: PredictionInput):
    """
    Make a prediction with feature importance explanation and full XCS.

    Computes the complete XCS formula in real-time:
        XCS = 0.4*Conf + 0.3*(1-SHAP_Instability) + 0.3*Jaccard(SHAP,LIME)

    Latency: ~200-500ms due to dual SHAP + LIME computation.
    Use /predict for fast-path (~10ms, confidence-only XCS).

    Parameters
    ----------
    input_data : PredictionInput
        Network flow features.

    Returns
    -------
    ExplanationOutput
        Prediction with SHAP feature importances, full XCS score,
        XCS component breakdown, and xcs_reliable flag.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")

    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    try:
        features = np.array(input_data.features, dtype=np.float64).reshape(1, -1)
        _validate_feature_count(features)
        features_scaled = scaler.transform(features)

        model = models.get("random_forest")
        if not model:
            model = models.get("xgboost")
        if not model:
            model = models.get("lightgbm")
        if not model:
            raise HTTPException(
                status_code=503, detail="No model loaded for explanation"
            )

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])

        try:
            import shap as shap_lib
            explainer = shap_lib.TreeExplainer(model)
            shap_vals = explainer.shap_values(features_scaled)
            # Handle multiclass (list or 3-D array) and binary (2-D array)
            if isinstance(shap_vals, list):
                sv = shap_vals[int(prediction)]
            elif shap_vals.ndim == 3:
                sv = shap_vals[0, :, int(prediction)]
            else:
                sv = shap_vals[0]
            abs_sv = [abs(float(v)) for v in sv]
            n_feats = len(abs_sv)
            feat_labels = (
                feature_names if feature_names and len(feature_names) == n_feats
                else [f"feature_{i}" for i in range(n_feats)]
            )
            pairs = sorted(
                zip(feat_labels, abs_sv),
                key=lambda x: x[1], reverse=True
            )
            top_features = [
                {"feature": f, "importance": round(imp, 6)}
                for f, imp in pairs[:10]
            ]
        except Exception as shap_err:
            logger.warning(f"SHAP failed, falling back to global importances: {shap_err}")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = abs(model.coef_[0])
            else:
                raise HTTPException(status_code=500, detail="Cannot compute explanation")
            n_feats = len(importances)
            feat_labels = (
                feature_names if feature_names and len(feature_names) == n_feats
                else [f"feature_{i}" for i in range(n_feats)]
            )
            pairs = sorted(
                zip(feat_labels, [float(v) for v in importances]),
                key=lambda x: x[1], reverse=True
            )
            top_features = [
                {"feature": f, "importance": round(imp, 6)}
                for f, imp in pairs[:10]
            ]

        # Full real-time XCS computation
        if feature_names and len(feature_names) == n_feats:
            fn = feature_names
        else:
            fn = [f"feature_{i}" for i in range(n_feats)]
        xcs_result = compute_realtime_xcs(model, features_scaled, fn, confidence)

        return ExplanationOutput(
            prediction=predicted_class,
            confidence=confidence,
            top_features=top_features,
            xcs_score=xcs_result["xcs_score"],
            xcs_components=xcs_result["components"],
            xcs_reliable=xcs_result["xcs_reliable"],
        )

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: expected {scaler.n_features_in_} features, got {len(input_data.features)}"
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Explanation error")
        raise HTTPException(status_code=500, detail="Explanation failed") from None


class BatchPredictionInput(BaseModel):
    features: List[List[float]] = Field(
        ...,
        description="List of network flow feature vectors",
        min_length=1,
        max_length=100,
    )

    @field_validator("features")
    @classmethod
    def reject_invalid_vectors(cls, values: List[List[float]]) -> List[List[float]]:
        if not values or any(not vector for vector in values):
            raise ValueError("features must contain non-empty vectors")
        width = len(values[0])
        if any(len(vector) != width for vector in values):
            raise ValueError("all feature vectors must have the same length")
        if not all(math.isfinite(value) for vector in values for value in vector):
            raise ValueError("features must contain only finite numbers")
        return values


class BatchPredictionOutput(BaseModel):
    predictions: List[str]
    confidences: List[float]
    xcs_scores: List[float]


@app.post("/predict-batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """Make predictions for a batch of network flows.

    Parameters
    ----------
    input_data : BatchPredictionInput
        List of feature vectors.

    Returns
    -------
    BatchPredictionOutput
        Batch predictions with confidences and fast-path XCS scores.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")
    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    try:
        features = np.array(input_data.features, dtype=np.float64)
        _validate_feature_count(features)
        features_scaled = scaler.transform(features)

        model = models.get("random_forest") or models.get("xgboost") or models.get("lightgbm")
        if not model:
            raise HTTPException(status_code=503, detail="No prediction model loaded")

        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)

        preds = [str(label_encoder.inverse_transform([p])[0]) for p in predictions]
        confs = [float(probabilities[i, predictions[i]]) for i in range(len(predictions))]
        xcs = [round(0.4 * c, 4) for c in confs]

        return BatchPredictionOutput(predictions=preds, confidences=confs, xcs_scores=xcs)

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch input")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail="Batch prediction failed") from None


@app.get("/classes")
async def get_classes():
    """Get list of available classes."""
    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Label encoder not loaded")

    return {"classes": [str(c) for c in label_encoder.classes_]}


@app.get("/xcs-summary")
async def xcs_summary():
    """Return XCS evaluation summary from offline computation."""
    import csv

    summary = {}
    for ds in ["CICIDS2017", "UNSWNB15", "CICIDS2018"]:
        for suffix in ["_v2", ""]:
            csv_path = Path(f"explanations/xcs_{ds}{suffix}.csv")
            if csv_path.exists():
                with open(csv_path) as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    xcs_vals = [float(r["xcs"]) for r in rows]
                    correct = [float(r["xcs"]) for r in rows if r["correct"] == "True"]
                    wrong = [float(r["xcs"]) for r in rows if r["correct"] == "False"]
                    flagged = sum(1 for r in rows if r["flag_review"] == "True")
                    summary[ds] = {
                        "n_samples": len(rows),
                        "mean_xcs": round(sum(xcs_vals) / len(xcs_vals), 4),
                        "flagged": flagged,
                        "flag_pct": round(flagged / len(rows) * 100, 1),
                        "correct_xcs": round(sum(correct) / len(correct), 4) if correct else None,
                        "wrong_xcs": round(sum(wrong) / len(wrong), 4) if wrong else None,
                        "source": str(csv_path),
                    }
                break

    return {
        "xcs_formula": "XCS = 0.4*Conf + 0.3*(1-SHAP_Instability) + 0.3*Jaccard(SHAP,LIME)",
        "threshold": 0.3,
        "interpretation": "XCS < 0.3 → flag for human analyst review",
        "datasets": summary,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
