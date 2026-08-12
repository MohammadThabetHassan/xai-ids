"""Tests for the FastAPI API endpoints."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the API."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api.app import app

    return TestClient(app)


class TestRootEndpoint:
    """Test GET / endpoint."""

    def test_root_returns_dict(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "models_loaded" in data

    def test_root_version(self, client):
        response = client.get("/")
        data = response.json()
        assert "version" in data


class TestHealthEndpoint:
    """Test GET /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_has_models_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert "models_loaded" in data
        assert "scaler_loaded" in data


class TestHealthFeaturesEndpoint:
    """Test GET /health/features endpoint."""

    def test_health_features_returns_200_or_503(self, client):
        response = client.get("/health/features")
        assert response.status_code in (200, 503)


class TestInputValidation:
    """Validate request boundaries before model execution."""

    def test_rejects_non_finite_features(self):
        from pydantic import ValidationError

        from api.app import PredictionInput

        with pytest.raises(ValidationError, match="finite"):
            PredictionInput(features=[0.0, float("nan")])

    def test_rejects_ragged_batch(self):
        from pydantic import ValidationError

        from api.app import BatchPredictionInput

        with pytest.raises(ValidationError, match="same length"):
            BatchPredictionInput(features=[[0.0, 1.0], [0.0]])

    def test_rejects_feature_count_before_scaler(self, monkeypatch):
        import numpy as np

        import api.app as api_app

        class Scaler:
            n_features_in_ = 3

        monkeypatch.setattr(api_app, "scaler", Scaler())
        with pytest.raises(api_app.HTTPException) as exc_info:
            api_app._validate_feature_count(np.zeros((1, 2)))
        assert exc_info.value.status_code == 400
        assert "expected 3 features" in exc_info.value.detail


class TestPredictEndpoint:
    """Test POST /predict endpoint."""

    def test_predict_requires_features(self, client):
        response = client.post("/predict", json={})
        assert response.status_code in (422, 503)

    def test_predict_with_valid_features(self, client):
        response = client.post(
            "/predict",
            json={"features": [0.0] * 20},
        )
        # Either succeeds (models loaded) or returns 503 (not loaded)
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
            assert "xcs_score" in data
            assert 0 <= data["confidence"] <= 1


class TestExplainEndpoint:
    """Test POST /explain endpoint."""

    def test_explain_with_valid_features(self, client):
        response = client.post(
            "/explain",
            json={"features": [0.0] * 20},
        )
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "top_features" in data
            assert "xcs_score" in data
            assert "xcs_components" in data
            assert "xcs_reliable" in data
            assert isinstance(data["xcs_reliable"], bool)
            assert len(data["top_features"]) <= 10


class TestClassesEndpoint:
    """Test GET /classes endpoint."""

    def test_classes_returns_list_or_503(self, client):
        response = client.get("/classes")
        assert response.status_code in (200, 503)

        if response.status_code == 200:
            data = response.json()
            assert "classes" in data
            assert isinstance(data["classes"], list)


class TestXcsSummaryEndpoint:
    """Test GET /xcs-summary endpoint."""

    def test_xcs_summary_returns_dict(self, client):
        response = client.get("/xcs-summary")
        assert response.status_code == 200
        data = response.json()
        assert "xcs_formula" in data
        assert "threshold" in data
        assert "datasets" in data

    def test_xcs_summary_threshold(self, client):
        response = client.get("/xcs-summary")
        data = response.json()
        assert data["threshold"] == 0.3
