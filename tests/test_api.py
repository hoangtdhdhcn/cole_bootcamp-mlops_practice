import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# ------------------------------------------------------------------
# Patch the model before importing the app
# ------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def mock_model_loading():
    """Mock pickle.load() and model.predict() so the real model file isn't needed."""
    fake_model = MagicMock()
    fake_model.predict.return_value = [0, 1, 2]  # Dummy outputs for testing

    # Patch pickle.load to return our fake model
    with patch("builtins.open", create=True), patch("pickle.load", return_value=fake_model):
        yield  # The app will import here with the mocked model

# ------------------------------------------------------------------
# Import the FastAPI app after mocking
# ------------------------------------------------------------------
from scripts.session_3.demo_api import app

client = TestClient(app)

# ------------------------------------------------------------------
# Test: /predict endpoint
# ------------------------------------------------------------------
def test_predict_endpoint():
    payload = [
        {"petal_length": 1.4, "petal_width": 0.2},
        {"petal_length": 4.7, "petal_width": 1.4},
        {"petal_length": 5.9, "petal_width": 2.1},
    ]

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 3
    assert all(species in ["setosa", "versicolor", "virginica"] for species in data["predictions"])

# ------------------------------------------------------------------
# Optional test: invalid input
# ------------------------------------------------------------------
def test_predict_invalid_input():
    # Missing one field should cause validation error
    bad_payload = [{"petal_length": 1.4}]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Pydantic validation error
