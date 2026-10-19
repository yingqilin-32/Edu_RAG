from copy import deepcopy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from edu_rag import rag, workflow, deployment
from edu_rag.config import Settings
from edu_rag.main import app


def plan(intent="knowledge_qa"):
    return {"status": "ready", "message": "", "origin": "rules", "intent": intent,
            "effective_scope": {"course_id": None, "grade": 3, "textbook_edition": "原创", "semester": None,
                                "subject": "math", "curriculum": None},
            "filters": {"grade": 3, "knowledge_point": "长方形面积"}, "effective_top_k": 5,
            "semantic_query": "长方形面积为什么长乘宽", "bm25_query": "长方形 面积", "requested_limit": None,
            "topic_filter": [{"id": "topic-1", "title": "长方形面积"}], "content_kind": "knowledge",
            "strategy": "filtered_retrieval", "usage": {}, "model": None}


def source(label="C1"):
    return {"id": "chunk-"+label, "citation": label, "source_title": "长方形面积知识卡",
            "content": "长方形面积=长×宽。用单位小正方形铺满图形。", "knowledge_point_id": "topic-1",
            "document_id": "doc-1", "exercise_id": None, "grade": 3, "subject": "math", "source_uri": "original://math"}


def retrieval(sources=None, scanned=8):
    return {"question": "长方形面积为什么长乘宽", "query_plan": {}, "sources": [source()] if sources is None else sources,
            "exercises": [], "embedding_model": "synthetic-test", "retrieval": {"mode": "bm25", "scanned_candidates": scanned,
            "candidate_k": 20, "rrf_k": 60, "candidate_lists": {"dense": [], "bm25": []},
            "timing_ms": {"total": 1, "planning": 0, "embedding": 0, "database": 1, "ranking": 0}}}


def judgement(*, answerable=True, quote="长方形面积=长×宽。", score=0.95, clarify=False):
    return {"scores": [{"citation": "C1", "relevance": score if answerable else 0.2, "quote": quote}],
            "requirements": [{"requirement": "解释长乘宽", "supported": answerable, "citations": ["C1"] if answerable else []}],
            "answerable": answerable, "needs_clarification": clarify,
            "clarification": "请补充完整尺寸。" if clarify else "", "reason": "证据支持" if answerable else "缺少要求的证据"}


def check(valid=True):
    return {"supported": valid, "addresses_question": valid, "citations_valid": valid, "reason": "核验完成"}


def runner(monkeypatch, replies, *, ready_plan=None, result=None, trace_file=None):
    ready_plan = ready_plan or plan()
    monkeypatch.setattr(rag, "construct_plan", lambda *args, **kwargs: deepcopy(ready_plan))
    requests = []
    def search(request, **kwargs):
        requests.append((request.model_dump(), deepcopy(kwargs["prepared_plan"])))
        return deepcopy(result or retrieval())
    monkeypatch.setattr(rag, "search", search)
    queue = list(replies)
    def respond(request):
        reply = queue.pop(0)
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": "secret-must-not-appear"})
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)}}],
            "usage": {"total_tokens": 30, "prompt_tokens": 20, "completion_tokens": 10}})
    client = httpx.Client(transport=httpx.MockTransport(respond))
    service = workflow.EvidenceWorkflow(settings=Settings(_env_file=None, deepseek_api_key="synthetic-test-key"),
                                        client=client, trace_file=trace_file)
    return service, queue, requests, client


def test_graph_generates_then_verifies_citations_and_persists_trace(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    service, queue, requests, client = runner(monkeypatch, [judgement(), "面积=长×宽。[C1]", check()], trace_file=path)
    with client:
        result = service.run(workflow.WorkflowRequest(question="长方形面积为什么长乘宽", query_mode="rules"))
    assert result["status"] == "answered"
    assert [e["node"] for e in result["workflow"]["events"]] == ["understand", "retrieve", "rerank", "evaluate", "generate", "verify", "finish"]
    assert result["workflow"]["retry_count"] == 0
    assert result["usage"]["total_tokens"] == 90
    trace = json.loads(path.read_text(encoding="utf-8"))
    assert trace["citation_sources"][0]["source_uri"] == "original://math"
    assert "synthetic-test-key" not in path.read_text(encoding="utf-8")
    assert not queue


def test_bounded_two_retries_preserve_scope_tags_and_original_facts(monkeypatch):
    replies = [judgement(answerable=False), {"additional_terms": ["单位方格"]},
               judgement(answerable=False), {"additional_terms": ["铺满方格"]}, judgement(answerable=False)]
    service, queue, requests, client = runner(monkeypatch, replies)
    with client:
        result = service.run(workflow.WorkflowRequest(question="长方形长8厘米宽3厘米，面积公式的历史是谁提出？", query_mode="rules"))
    assert result["status"] == "refused"
    assert result["workflow"]["retry_count"] == 2
    assert result["workflow"]["retrieval_count"] == 3
    assert len(requests) == 3 and not queue
    for request, planned in requests:
        assert planned["effective_scope"] == plan()["effective_scope"]
        assert planned["filters"] == plan()["filters"]
        assert planned["topic_filter"] == plan()["topic_filter"]
    assert "8厘米宽3厘米" in requests[1][1]["semantic_query"]
    assert requests[1][0]["retrieval_mode"] == "hybrid"
    assert result["sources"] == [] and result["citations"] == []


def test_retrieval_can_recover_on_first_retry(monkeypatch):
    service, queue, requests, client = runner(monkeypatch, [judgement(answerable=False), {"additional_terms": ["方格铺满"]},
                                                           judgement(), "面积=长×宽。[C1]", check()])
    with client:
        result = service.run(workflow.WorkflowRequest(question="长方形面积为什么长乘宽", query_mode="rules"))
    assert result["status"] == "answered" and result["workflow"]["retry_count"] == 1
    assert not queue


@pytest.mark.parametrize("bad_grade", [judgement(quote="不存在的原文"), judgement(score=0.74),
                                      {**judgement(), "requirements": [{"requirement": "未知事实", "supported": True, "citations": ["C9"]}]}])
def test_unsupported_quotes_scores_or_requirement_citations_never_pass(monkeypatch, bad_grade):
    service, queue, requests, client = runner(monkeypatch, [bad_grade])
    with client:
        result = service.run(workflow.WorkflowRequest(question="长方形面积为什么长乘宽", query_mode="rules", max_retries=0))
    assert result["status"] == "refused" and result["sources"] == []


@pytest.mark.parametrize("reply", [402, "not-json", {**judgement(), "scores": [{"citation": "C9", "relevance": 1, "quote": "长方形"}]}])
def test_model_outage_or_malformed_grades_fail_closed_and_record_error(monkeypatch, reply):
    service, queue, requests, client = runner(monkeypatch, [reply])
    with client:
        result = service.run(workflow.WorkflowRequest(question="长方形面积为什么长乘宽", query_mode="rules"))
    assert result["status"] == "refused" and result["workflow"]["error"]
    assert "secret-must-not-appear" not in json.dumps(result)
    assert result["workflow"]["retrieval_count"] == 1


@pytest.mark.parametrize("draft,verification", [("面积是99。[C1]", check(False)), ("面积是99。[C9]", None)])
def test_invalid_or_ungrounded_answer_draft_is_suppressed(monkeypatch, draft, verification):
    replies = [judgement(), draft] + ([verification] if verification else [])
    service, queue, requests, client = runner(monkeypatch, replies)
    with client:
        result = service.run(workflow.WorkflowRequest(question="长方形面积为什么长乘宽", query_mode="rules"))
    assert result["status"] == "refused"
    assert "99" not in result["answer"] and result["citations"] == []
    assert "answer_result" not in result


def test_missing_input_clarifies_without_generation_or_retry(monkeypatch):
    service, queue, requests, client = runner(monkeypatch, [judgement(answerable=False, clarify=True)])
    with client:
        result = service.run(workflow.WorkflowRequest(question="求这个长方形的面积", query_mode="rules"))
    assert result["status"] == "needs_clarification" and result["workflow"]["retry_count"] == 0
    assert not queue


def test_empty_database_result_does_not_waste_model_or_retries(monkeypatch):
    service, queue, requests, client = runner(monkeypatch, [], result=retrieval(sources=[], scanned=0))
    with client:
        result = service.run(workflow.WorkflowRequest(question="未收录主题是什么", query_mode="rules"))
    assert result["status"] == "refused" and result["workflow"]["calls"] == []
    assert result["workflow"]["retrieval_count"] == 1


@pytest.mark.parametrize("status,expected", [("no_matching_scope", "refused"), ("needs_clarification", "needs_clarification")])
def test_preflight_scope_failure_stops_before_retrieval(monkeypatch, status, expected):
    ready_plan = {**plan(), "status": status, "message": "请调整范围"}
    service, queue, requests, client = runner(monkeypatch, [], ready_plan=ready_plan)
    with client:
        result = service.run(workflow.WorkflowRequest(question="初三一元二次方程", query_mode="rules"))
    assert result["status"] == expected and not requests


def test_preview_runs_evidence_gate_without_generating_answer(monkeypatch):
    service, queue, requests, client = runner(monkeypatch, [judgement()])
    with client:
        result = service.run(workflow.WorkflowRequest(question="长方形面积为什么长乘宽", query_mode="rules"), preview=True)
    assert result["status"] == "evidence_ready"
    assert [c["stage"] for c in result["workflow"]["calls"]] == ["rerank"]


def test_numeric_rewrite_is_rejected_before_new_retrieval(monkeypatch):
    service, queue, requests, client = runner(monkeypatch, [judgement(answerable=False), {"additional_terms": ["长9米"]}])
    with client:
        result = service.run(workflow.WorkflowRequest(question="长方形长8米面积怎么算", query_mode="rules"))
    assert result["status"] == "refused" and len(requests) == 1


def test_summary_budget_shortfall_requests_narrower_scope(monkeypatch):
    ready_plan = plan("knowledge_summary")
    result = {**retrieval(), "query_plan": {"coverage": {"catalog_topics": 16, "partial": True}}}
    service, queue, requests, client = runner(monkeypatch, [judgement()], ready_plan=ready_plan, result=result)
    with client:
        response = service.run(workflow.WorkflowRequest(question="总结整年级知识", query_mode="rules", summary_max_sources=1))
    assert response["status"] == "needs_clarification"


@pytest.mark.parametrize("body", [{"question": "面积怎么算", "max_retries": 3},
                                  {"question": "面积怎么算", "max_retries": True},
                                  {"question": "面积怎么算", "query_mode": "off"}])
def test_v4_rejects_unbounded_retries_and_disabled_understanding(body):
    with TestClient(app) as client:
        assert client.post("/api/v4/ask", json=body).status_code == 422


def test_v4_endpoint_uses_workflow_and_keeps_v3_available(monkeypatch):
    monkeypatch.setattr(deployment, "ask", lambda r: {"status": "refused", "question": r.question})
    with TestClient(app) as client:
        response = client.post("/api/v4/ask", json={"question": "资料够吗"})
        assert response.status_code == 200 and response.json()["status"] == "refused"
        assert "/api/v3/ask" in client.get("/openapi.json").json()["paths"]
