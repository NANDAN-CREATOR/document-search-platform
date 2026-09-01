import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    with patch("tracing.phoenix_setup.instrument_all"):
        from api.main import app
        return TestClient(app)

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Document Search Platform"

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_search_empty_query(client):
    response = client.post("/api/v1/search", json={"query": ""})
    assert response.status_code == 400

def test_ingest_status(client):
    response = client.get("/api/v1/ingest/status")
    assert response.status_code == 200
