import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


@pytest.mark.parametrize("size", [1, 50])
def test_valid_batch(client, size):
    response = client.post(
        "/predict_batch",
        json={"comments": ["This video is amazing!"] * size},
    )

    assert response.status_code == 200
    data = response.json()

    assert len(data["results"]) == size
    assert data["statistics"]["total"] == size
    assert sum(data["statistics"]["counts"].values()) == size

    for result in data["results"]:
        assert result["label"] in [-1, 0, 1]
        assert 0 <= result["confidence"] <= 1


@pytest.mark.parametrize("payload", [
    {"comments": []},
    {"comments": [""]},
    {"comments": ["   "]},
    {"comments": ["!!!"]},
    {"comments": ["hello"] * 51},
    {"comments": ["a" * 20001]},
    {"comments": [123]},
    {},
])
def test_invalid_input(client, payload):
    response = client.post("/predict_batch", json=payload)

    assert response.status_code == 422