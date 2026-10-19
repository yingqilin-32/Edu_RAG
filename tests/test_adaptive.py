from copy import deepcopy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from edu_rag import rag
from edu_rag.adaptive import AdaptiveRequest, AdaptiveWorkflow
from edu_rag.budget import BudgetClient, BudgetExceeded
from edu_rag.config import Settings
from edu_rag.main import app
from test_gating import review
from test_workflow import plan, retrieval, source, check


def gap(kind="retrieval_gap", supported=False):
    return {**review(supported=supported), "gap_type": kind}


def scores(sources):
    return {"scores": [{"citation": s["citation"], "relevance": .95, "quote": "长方形面积"} for s in sources]}


def runner(monkeypatch, replies, *, results=None, policy="adaptive", ready=None):
    monkeypatch.setattr(rag, "construct_plan", lambda *a, **kw: deepcopy(ready or plan()))
    pools, searches, bodies, queue = list(results or [retrieval()]), [], [], list(replies)
    def search(request, **kwargs):
        searches.append(deepcopy(kwargs["prepared_plan"]))
        return deepcopy(pools.pop(0) if len(pools) > 1 else pools[0])
    monkeypatch.setattr(rag, "search", search)
    def respond(request):
        bodies.append(json.loads(request.content))
        reply = queue.pop(0)
        if isinstance(reply, int): return httpx.Response(reply)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})
    client = httpx.Client(transport=httpx.MockTransport(respond))
    service = AdaptiveWorkflow(retry_policy=policy, settings=Settings(_env_file=None, deepseek_api_key="fake"), client=client, trace_file=None)
    return service, client, queue, searches, bodies


def test_success_uses_single_path_and_tracks_actual_provider_cost(monkeypatch):
    service, client, queue, searches, _ = runner(monkeypatch, [scores([source()]), gap("none", True), "面积=长×宽。[C1]", check()])
    with client: result = service.run(AdaptiveRequest(question="长方形面积怎么算", query_mode="rules"))
    assert result["status"] == "answered" and not queue
    assert len(searches) == 1 and result["workflow"]["branch_history"] == []
    assert result["workflow"]["budget"]["model_calls"] == 4 and result["usage"]["total_tokens"] == 120
    assert result["workflow"]["stop_reason"] == "answered"


def test_known_missing_knowledge_skips_rewrite(monkeypatch):
    service, client, queue, searches, _ = runner(monkeypatch, [scores([source()]), gap("missing_knowledge")])
    with client: result = service.run(AdaptiveRequest(question="面积公式作者的身份证号码？"))
    assert result["status"] == "refused" and not queue and len(searches) == 1
    assert result["workflow"]["stop_reason"] == "missing_knowledge"
    assert result["workflow"]["budget"]["model_calls"] == 2


def test_rewrite_branches_recover_missing_evidence_with_stable_scope_and_citations(monkeypatch):
    new = {**source("C1"), "id": "new-chunk", "document_id": "new-doc"}
    service, client, queue, searches, _ = runner(monkeypatch,
        [scores([source()]), gap(), {"additional_terms": ["统一单位", "铺满方格"]},
         scores([source(), {**new, "citation": "C2"}]), gap("none", True)],
        results=[retrieval(), retrieval([source(), new]), retrieval([new])])
    with client: result = service.run(AdaptiveRequest(question="长方形长8厘米，需要哪些方法？", max_branches=2), preview=True)
    assert result["status"] == "evidence_ready" and not queue
    assert result["workflow"]["retrieval_count"] == 3 and result["workflow"]["retry_count"] == 1
    assert len(result["workflow"]["branch_history"]) == 2
    assert {s["id"]: s["citation"] for s in result["sources"]} == {"chunk-C1": "C1", "new-chunk": "C2"}
    assert all(p["effective_scope"] == plan()["effective_scope"] and p["filters"] == plan()["filters"] and p["topic_filter"] == plan()["topic_filter"] for p in searches)
    assert all("长8厘米" in p["semantic_query"] for p in searches[1:])


def test_unchanged_candidates_stop_before_repeated_scoring(monkeypatch):
    service, client, queue, searches, _ = runner(monkeypatch, [scores([source()]), gap(), {"additional_terms": ["统一单位", "面积公式"]}])
    with client: result = service.run(AdaptiveRequest(question="长方形面积需要哪些方法？"))
    assert result["status"] == "refused" and not queue
    assert result["workflow"]["stop_reason"] == "no_new_evidence"
    assert result["workflow"]["budget"]["model_calls"] == 3
    assert len(searches) == 3


def test_request_call_cap_blocks_at_http_boundary(monkeypatch):
    service, client, queue, _, bodies = runner(monkeypatch, [scores([source()])])
    with client: result = service.run(AdaptiveRequest(question="长方形面积怎么算", max_model_calls=1))
    assert result["status"] == "budget_exceeded" and len(bodies) == 1 and not queue
    assert result["workflow"]["budget"]["denied_reason"] == "model_call_limit"
    assert result["sources"] == [] and result["citations"] == []


def test_retry_reserves_generation_and_verification_calls(monkeypatch):
    service, client, queue, searches, _ = runner(monkeypatch, [scores([source()]), gap()])
    with client: result = service.run(AdaptiveRequest(question="长方形面积怎么算", max_model_calls=6))
    assert result["status"] == "budget_exceeded" and not queue
    assert result["workflow"]["stop_reason"] == "reserve_final_calls" and len(searches) == 1


def test_rewrite_cannot_introduce_new_numeric_conditions(monkeypatch):
    service, client, queue, searches, _ = runner(monkeypatch, [scores([source()]), gap(), {"additional_terms": ["长9米"]}])
    with client: result = service.run(AdaptiveRequest(question="长方形长8厘米，面积怎么算"))
    assert result["status"] == "refused" and not queue and len(searches) == 1
    assert result["workflow"]["stop_reason"] == "no_new_query"


def test_low_token_budget_rejects_before_sending_request(monkeypatch):
    service, client, queue, _, bodies = runner(monkeypatch, [])
    with client: result = service.run(AdaptiveRequest(question="长方形面积怎么算", max_total_tokens=1000))
    assert result["status"] == "budget_exceeded" and not bodies
    assert result["workflow"]["budget"]["denied_reason"] == "token_admission_limit"


def test_model_deadline_prevents_any_network_call():
    with httpx.Client(transport=httpx.MockTransport(lambda r: pytest.fail("network call after deadline"))) as client:
        meter = BudgetClient(max_calls=2, max_tokens=30000, max_seconds=0, timeout=10, client=client)
        with pytest.raises(BudgetExceeded): meter.post("https://example.test", headers={}, json={"messages": [], "max_tokens": 10})
    assert meter.denied == "model_deadline" and not meter.records


@pytest.mark.parametrize("body", [{"max_model_calls": 13}, {"max_retries": 3}, {"max_branches": 4}, {"max_model_calls": True}, {"max_total_tokens": 999}, {"bypass_gate": True}])
def test_public_api_rejects_unbounded_or_gate_bypass_options(body):
    with TestClient(app) as client:
        assert client.post("/api/v5/ask", json={"question": "长方形面积怎么算", **body}).status_code == 422


def test_single_pass_is_a_true_no_rewrite_control(monkeypatch):
    service, client, queue, searches, _ = runner(monkeypatch, [scores([source()]), gap()], policy="single_pass")
    with client: result = service.run(AdaptiveRequest(question="长方形面积怎么算"))
    assert result["status"] == "refused" and len(searches) == 1 and not queue
    assert result["workflow"]["stop_reason"] == "single_pass"


def test_always_retry_control_respects_two_retry_bound(monkeypatch):
    service, client, queue, searches, _ = runner(monkeypatch,
        [scores([source()]), gap(), {"additional_terms": ["统一单位"]}, scores([source()]), gap(),
         {"additional_terms": ["铺满方格"]}, scores([source()]), gap()], policy="always_retry")
    with client: result = service.run(AdaptiveRequest(question="长方形面积怎么算"), preview=True)
    assert result["status"] == "refused" and not queue and len(searches) == 3
    assert result["workflow"]["retry_count"] == 2 and result["workflow"]["budget"]["model_calls"] == 8


def test_missing_input_exits_before_model_or_retrieval(monkeypatch):
    service, client, queue, searches, bodies = runner(monkeypatch, [], ready={**plan(), "status": "needs_clarification", "message": "请补充宽。"})
    with client: result = service.run(AdaptiveRequest(question="长方形长9厘米，面积是多少"))
    assert result["status"] == "needs_clarification" and not searches and not bodies


def test_budget_denial_in_planner_is_not_hidden_by_rule_fallback(monkeypatch):
    service, client, queue, searches, bodies = runner(monkeypatch, [])
    def fallback(request, **kwargs):
        try:
            kwargs["client"].post("https://example.test", headers={}, json={"messages": [], "max_tokens": 2000})
        except BudgetExceeded:
            return plan()
    monkeypatch.setattr(rag, "construct_plan", fallback)
    with client: result = service.run(AdaptiveRequest(question="长方形面积怎么算", max_total_tokens=1000))
    assert result["status"] == "budget_exceeded" and not searches and not bodies
