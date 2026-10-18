from fastapi.testclient import TestClient
import pytest

from edu_rag.main import app
from edu_rag.scope import CurriculumScope


@pytest.mark.parametrize("body", [
    {"grade": 0}, {"grade": 13}, {"textbook_edition": "  "},
    {"semester": "unknown"}, {"subject": "unknown"}, {"curriculum": " "},
])
def test_invalid_scope_fails_before_database_or_inference(body):
    with TestClient(app) as client:
        assert client.get("/api/v2/courses", params=body).status_code == 422
        assert client.post("/api/v2/search", json={"question": "解释面积", **body}).status_code == 422


def test_catalog_receives_the_same_exact_scope(monkeypatch):
    scopes = []
    monkeypatch.setattr("edu_rag.main.get_engine", lambda: object())
    monkeypatch.setattr("edu_rag.knowledge.courses", lambda engine, scope: scopes.append(scope) or [])
    with TestClient(app) as client:
        response = client.get("/api/v2/courses", params={"grade": 3, "textbook_edition": "自编演示资料", "semester": "full_year"})
    assert response.status_code == 200
    assert scopes == [CurriculumScope(grade=3, textbook_edition="自编演示资料", semester="full_year")]
