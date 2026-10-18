import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from edu_rag.config import Settings
from edu_rag.ingest import Dataset, load_dataset
from edu_rag.main import app
from edu_rag import rag


def source(label="C1"):
    return {"citation": label, "source_title": "长方形面积", "content": "长方形面积=长×宽。"}


def model_client(answer, status=200, check_request=None):
    def respond(request):
        assert request.url.path == "/chat/completions"
        if check_request is not None:
            check_request(request)
        return httpx.Response(status, json={
            "choices": [{"finish_reason": "stop", "message": {"content": answer}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        })
    return httpx.Client(transport=httpx.MockTransport(respond))


def settings():
    return Settings(_env_file=None, deepseek_api_key="synthetic-test-key")


def test_citations_resolve_to_retrieved_sources():
    with model_client("面积为 8×3=24 平方厘米。[C1]") as client:
        result = rag.generate("求面积", [source()], settings=settings(), client=client)
    assert result["status"] == "answered"
    assert result["citations"] == [source()]
    assert result["usage"]["total_tokens"] == 30


def test_retrieved_exercise_solution_reaches_answer_generation(monkeypatch):
    row = {"id": "chunk-1", "similarity": 0.9, "exercise_id": "exercise-1", "document_id": None,
           "grade": 4, "subject": "math", "textbook_edition": "原创测试版", "semester": "first",
           "source_title": "周长练习", "content": "长方形长8厘米、宽3厘米，周长是多少？",
           "exercise_answer": "22厘米", "exercise_explanation": "周长=(长+宽)×2=(8+3)×2=22厘米。"}
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    connection.execute.return_value.mappings.return_value.all.return_value = [row]
    monkeypatch.setattr(rag, "get_engine", lambda: engine)
    monkeypatch.setattr(rag, "get_encoder", lambda: SimpleNamespace(model_id="test", encode=lambda texts: [[1.0]]))

    def check_evidence(request):
        body = json.loads(request.content)
        evidence = json.loads(body["messages"][1]["content"])["sources"][0]
        assert row["content"] in evidence["content"]
        assert row["exercise_answer"] in evidence["content"]
        assert row["exercise_explanation"] in evidence["content"]
        assert evidence["citation"] == "C1"
        assert evidence["course_scope"]["grade"] == 4
        assert evidence["course_scope"]["textbook_edition"] == "原创测试版"
        assert "适合小学三年级学生" not in body["messages"][0]["content"]

    generate = rag.generate
    with model_client("周长=(8+3)×2=22厘米。[C1]", check_request=check_evidence) as client:
        monkeypatch.setattr(rag, "generate", lambda question, sources: generate(question, sources, settings=settings(), client=client))
        result = rag.ask(rag.SearchRequest(question="求周长"))
    assert result["status"] == "answered"
    assert row["exercise_explanation"] in result["citations"][0]["content"]


@pytest.mark.parametrize("citation", ["C9", "C0"])
def test_fabricated_citation_is_rejected(citation):
    with model_client(f"面积是 24。[{citation}]") as client:
        with pytest.raises(rag.ModelError, match="不存在"):
            rag.generate("求面积", [source()], settings=settings(), client=client)


def test_uncited_answer_is_not_presented_as_grounded():
    with model_client("月球表面重力是地球的六分之一。") as client:
        result = rag.generate("月球重力是多少", [source()], settings=settings(), client=client)
    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == []
    assert "当前资料不足" in result["answer"]


def test_provider_error_does_not_echo_response_or_key():
    with model_client("synthetic-test-key", status=401) as client:
        with pytest.raises(rag.ModelError) as caught:
            rag.generate("求面积", [source()], settings=settings(), client=client)
    assert "未授权" in str(caught.value)
    assert "synthetic-test-key" not in str(caught.value)


def test_no_evidence_does_not_call_paid_model(monkeypatch):
    monkeypatch.setattr(rag, "search", lambda request: {"question": request.question, "sources": [], "embedding_model": "test"})
    def forbidden(*args, **kwargs):
        pytest.fail("No evidence must not trigger a model request")
    monkeypatch.setattr(rag, "generate", forbidden)
    assert rag.ask(rag.SearchRequest(question="解释面积"))["model"] is None


@pytest.mark.parametrize("body", [
    {"question": "  "}, {"question": "求面积", "top_k": 0},
    {"question": "求面积", "top_k": 11}, {"question": "求面积", "course_id": "not-a-uuid"},
])
def test_invalid_query_is_rejected_before_inference(body):
    with TestClient(app) as client:
        assert client.post("/api/v1/ask", json=body).status_code == 422


def test_root_explains_and_exposes_knowledge_browser():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "知识目录" in response.text
    assert "尚未覆盖完整教材" in response.text


def test_duplicate_topic_codes_are_rejected():
    data = load_dataset().model_dump()
    data["topics"].append(data["topics"][0])
    with pytest.raises(ValidationError, match="不能重复"):
        Dataset.model_validate(data)
