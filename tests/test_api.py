from fastapi.testclient import TestClient

from edu_rag.config import ConfigurationError
from edu_rag.main import app


def test_liveness_does_not_require_a_database():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["stage"] == "V3"


def test_database_failure_is_not_reported_as_healthy(monkeypatch):
    def missing_database():
        raise ConfigurationError("private-password-must-not-appear")

    monkeypatch.setattr("edu_rag.main.get_engine", missing_database)
    with TestClient(app) as client:
        response = client.get("/health/db")
        assert response.status_code == 503
        assert "private-password" not in response.text

