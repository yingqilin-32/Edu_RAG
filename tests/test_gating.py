from copy import deepcopy
import json

import httpx
import pytest

from edu_rag import rag
from edu_rag.config import Settings
from edu_rag.gating import GatedWorkflow
from edu_rag.workflow import WorkflowRequest
from test_workflow import plan, source, retrieval, check


def review(*, supported=True, clarify=False, citation="C1"):
    return {"requirements": [{"requirement": "面积公式", "supported": supported,
                               "citations": [citation] if supported else []}],
            "answerable": supported, "needs_clarification": clarify,
            "clarification": "请补充长和宽。" if clarify else "", "reason": "评估结果"}


def service(monkeypatch, replies, *, policy="retrieval_order", sources=None, ready_plan=None):
    monkeypatch.setattr(rag, "construct_plan", lambda *a, **kw: deepcopy(ready_plan or plan()))
    searches = []
    def search(request, **kwargs):
        searches.append(deepcopy(kwargs["prepared_plan"]))
        return retrieval(sources)
    monkeypatch.setattr(rag, "search", search)
    queue, bodies = list(replies), []
    def respond(request):
        bodies.append(json.loads(request.content))
        reply = queue.pop(0)
        if isinstance(reply, int): return httpx.Response(reply)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)}}], "usage": {"total_tokens": 10}})
    client = httpx.Client(transport=httpx.MockTransport(respond))
    runner = GatedWorkflow(context_policy=policy, settings=Settings(_env_file=None, deepseek_api_key="fake"),
                           client=client, trace_file=None)
    return runner, client, queue, bodies, searches


def test_rrf_order_has_independent_gate_and_no_ranker(monkeypatch):
    runner, client, queue, bodies, _ = service(monkeypatch, [review(), "面积=长×宽。[C1]", check()])
    with client:
        response = runner.run(WorkflowRequest(question="长方形面积怎么算", max_retries=0))
    assert response["status"] == "answered" and not queue
    assert [c["stage"] for c in response["workflow"]["calls"]] == ["gate", "generate", "verify"]
    assert [e["node"] for e in response["workflow"]["events"]] == ["understand", "retrieve", "select_context", "gate", "generate", "verify", "finish"]
    assert response["workflow"]["assessment"]["relevance_threshold"] is None
    assert response["workflow"]["version"] == "independent-evidence-gate-v1"


@pytest.mark.parametrize("policy,expected", [("retrieval_order", ["C1", "C2"]), ("llm_rerank", ["C2", "C1"]),
                                             ("score_filter", ["C2"]), ("llm_rerank_filter", ["C2"])])
def test_rank_order_and_score_filter_are_independent(monkeypatch, policy, expected):
    scores = {"scores": [{"citation": "C1", "relevance": .2, "quote": "长方形面积"},
                         {"citation": "C2", "relevance": .9, "quote": "长方形面积"}]}
    replies = ([] if policy == "retrieval_order" else [scores]) + [review(citation="C2")]
    runner, client, queue, _, _ = service(monkeypatch, replies, policy=policy, sources=[source(), source("C2")])
    with client:
        response = runner.run(WorkflowRequest(question="面积怎么算", max_retries=0), preview=True)
    assert response["status"] == "evidence_ready"
    assert [s["citation"] for s in response["sources"]] == expected and not queue
    assert response["workflow"]["calls"][-1]["stage"] == "gate"


def test_missing_user_input_clarifies_without_draft_or_retry(monkeypatch):
    runner, client, queue, _, searches = service(monkeypatch, [review(supported=False, clarify=True)])
    with client: response = runner.run(WorkflowRequest(question="这块纸的面积是多少？"))
    assert response["status"] == "needs_clarification" and "长和宽" in response["answer"]
    assert len(searches) == 1 and response["workflow"]["retry_count"] == 0
    assert [c["stage"] for c in response["workflow"]["calls"]] == ["gate"] and not queue


def test_missing_knowledge_retries_twice_without_changing_scope(monkeypatch):
    runner, client, queue, _, searches = service(monkeypatch, [review(supported=False), {"additional_terms": ["铺满方格"]},
                                                              review(supported=False), {"additional_terms": ["单位方格"]}, review(supported=False)])
    with client: response = runner.run(WorkflowRequest(question="长方形面积公式由谁最早提出？"))
    assert response["status"] == "refused" and len(searches) == 3
    assert response["workflow"]["retry_count"] == 2 and not queue
    assert all(s["effective_scope"] == plan()["effective_scope"] and s["filters"] == plan()["filters"] for s in searches)
    assert all("长方形面积公式由谁最早提出" in s["bm25_query"] for s in searches[1:])


@pytest.mark.parametrize("bad", [review(citation="C99"), {**review(), "requirements": []}, 402])
def test_invalid_evidence_or_provider_error_never_reaches_generation(monkeypatch, bad):
    runner, client, _, bodies, _ = service(monkeypatch, [bad])
    with client: response = runner.run(WorkflowRequest(question="面积怎么算", max_retries=0))
    assert response["status"] == "refused" and response["citations"] == []
    assert len(bodies) == 1
    assert bool(response["workflow"]["error"]) == isinstance(bad, int)


def test_unquoted_high_score_is_excluded_only_by_filter(monkeypatch):
    scores = {"scores": [{"citation": "C1", "relevance": 1, "quote": "伪造摘录"}]}
    runner, client, queue, _, _ = service(monkeypatch, [scores], policy="llm_rerank_filter")
    with client: response = runner.run(WorkflowRequest(question="面积怎么算", max_retries=0))
    assert response["status"] == "refused" and not queue
    assert response["workflow"]["searches"][0]["context_selection"]["selected_ids"] == []


def test_empty_pool_uses_no_model_calls(monkeypatch):
    runner, client, queue, bodies, _ = service(monkeypatch, [], sources=[])
    with client: response = runner.run(WorkflowRequest(question="尚未收录的定理", max_retries=0))
    assert response["status"] == "refused" and not bodies


def test_preflight_scope_stops_before_retrieval(monkeypatch):
    runner, client, _, bodies, searches = service(monkeypatch, [], ready_plan={**plan(), "status": "no_matching_scope"})
    with client: response = runner.run(WorkflowRequest(question="初三一元二次方程"))
    assert response["status"] == "refused" and not searches and not bodies


def test_gate_evaluates_only_selected_context_not_full_pool(monkeypatch):
    runner, client, _, bodies, _ = service(monkeypatch, [review(citation="C2")], sources=[source(), source("C2")],
                                          ready_plan={**plan(), "effective_top_k": 1})
    with client: response = runner.run(WorkflowRequest(question="面积怎么算", max_retries=0), preview=True)
    payload = json.loads(bodies[0]["messages"][1]["content"])
    assert [s["citation"] for s in payload["sources"]] == ["C1"]
    assert response["status"] == "refused"
