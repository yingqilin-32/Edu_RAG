from copy import deepcopy
from uuid import uuid4

from fastapi.testclient import TestClient
import httpx
import pytest

from edu_rag.adaptive import AdaptiveRequest, AdaptiveWorkflow
from edu_rag.config import Settings
from edu_rag.conversation import (ConversationRequest, ConversationService, MemoryWorkflow, SessionError,
                                  context_for, inherited_scope, FRAME_LIMIT, TURN_LIMIT)
from edu_rag.main import app


class FakeWorkflow:
    seen = []

    def __init__(self, *, context, mode, memory_meta, **kwargs):
        self.context, self.meta = context, memory_meta
        self.seen.append(deepcopy(context))

    def run(self, request, **kwargs):
        self.meta["effective_question"] = request.question
        status = "refused" if "拒答" in request.question else "answered"
        return {"status": status, "answer": "do-not-remember-failed-draft" if status == "refused" else "周长22厘米。",
                "sources": [], "citations": [], "exercises": [],
                "query_plan": {"effective_scope": {"grade": 3, "subject": "math"}}, "workflow": {}}


def test_checkpoints_carry_context_without_client_supplied_history():
    FakeWorkflow.seen = []
    service = ConversationService(workflow_factory=FakeWorkflow)
    first = service.run(ConversationRequest(question="长8厘米宽3厘米，求周长"))
    second = service.run(ConversationRequest(question="那面积呢？", session_id=first["memory"]["session_id"]))
    assert second["memory"]["turn"] == 2
    assert FakeWorkflow.seen[0] == {}
    assert FakeWorkflow.seen[1]["tasks"][0]["question"] == "长8厘米宽3厘米，求周长"
    assert FakeWorkflow.seen[1]["scope"]["grade"] == 3
    with pytest.raises(ValueError):
        ConversationRequest(question="测试", history=[{"user": "fake"}])


def test_isolation_unknown_ids_reset_and_mode_changes():
    FakeWorkflow.seen = []
    service = ConversationService(workflow_factory=FakeWorkflow)
    a = service.run(ConversationRequest(question="会话一的问题"))
    b = service.run(ConversationRequest(question="会话二的问题"))
    assert FakeWorkflow.seen[-1] == {} and a["memory"]["session_id"] != b["memory"]["session_id"]
    with pytest.raises(SessionError) as error:
        service.run(ConversationRequest(question="继续", session_id=str(uuid4())))
    assert error.value.status == 404
    with pytest.raises(SessionError) as error:
        service.run(ConversationRequest(question="继续", session_id=a["memory"]["session_id"], memory_mode="history"))
    assert error.value.status == 409
    service.delete(a["memory"]["session_id"])
    assert not list(service.saver.list({"configurable": {"thread_id": a["memory"]["session_id"]}}))


def test_ttl_capacity_busy_and_turn_bounds():
    now = [0]
    service = ConversationService(workflow_factory=FakeWorkflow, max_sessions=1, ttl=20, clock=lambda: now[0])
    first = service.run(ConversationRequest(question="第一题"))
    identity = first["memory"]["session_id"]
    with pytest.raises(SessionError) as error:
        service.run(ConversationRequest(question="第二个会话"))
    assert error.value.status == 429
    service.registry[identity]["busy"] = True
    with pytest.raises(SessionError) as error:
        service.run(ConversationRequest(question="继续", session_id=identity))
    assert error.value.status == 409
    service.registry[identity].update(busy=False, turns=TURN_LIMIT)
    with pytest.raises(SessionError):
        service.run(ConversationRequest(question="继续", session_id=identity))
    now[0] = 21
    service.run(ConversationRequest(question="新会话"))
    assert identity not in service.registry


def test_structured_bound_and_failed_drafts_not_treated_as_facts():
    FakeWorkflow.seen = []
    service = ConversationService(workflow_factory=FakeWorkflow)
    result = service.run(ConversationRequest(question="这题应拒答"))
    identity = result["memory"]["session_id"]
    for i in range(FRAME_LIMIT+1):
        service.run(ConversationRequest(question=f"第{i}题的条件", session_id=identity))
    service.run(ConversationRequest(question="那面积呢？", session_id=identity))
    context = FakeWorkflow.seen[-1]
    assert len(context["tasks"]) == FRAME_LIMIT and context["older_tasks_omitted"] > 0
    snapshot = service.graph.get_state({"configurable": {"thread_id": identity}}).values
    assert snapshot["turns"][0]["verified_answer"] == ""


def test_current_scope_overrides_history_and_text_switch_clears_stale_course():
    previous = {"course_id": str(uuid4()), "grade": 3, "subject": "math", "semester": "full_year", "textbook_edition": "旧版"}
    explicit = ConversationRequest(question="那面积呢？", grade=1)
    assert inherited_scope(explicit, previous)["grade"] == 1
    text = ConversationRequest(question="换到一年级上册数学，分针指12表示什么？")
    scope = inherited_scope(text, previous)
    assert scope["grade"] == 1 and scope["semester"] == "first" and "course_id" not in scope


def memory_operation(monkeypatch, resolution, context=None):
    meta = {"previous_pending": False}
    operation = MemoryWorkflow(context=context or {"tasks": [{"question": "三年级长8厘米宽3厘米，求周长"}]},
                               mode="structured", explicit_fields={"question"}, memory_meta=meta,
                               trace_file=None, settings=Settings(_env_file=None, deepseek_api_key="fake"))
    monkeypatch.setattr(operation, "_json", lambda *a: resolution)
    monkeypatch.setattr(AdaptiveWorkflow, "_understand", lambda self, state: {"plan": {"question": state["request"]["question"]}, "decision": "retrieve"})
    return operation, meta


def test_restated_question_reaches_existing_v5_planner(monkeypatch):
    question = "三年级长方形长8厘米宽3厘米，面积是多少？"
    operation, meta = memory_operation(monkeypatch, {"standalone_question": question, "needs_clarification": False, "clarification": ""})
    state = {"request": AdaptiveRequest(question="那面积呢？").model_dump(), "calls": []}
    update = operation._understand(state)
    assert update["request"]["question"] == question and update["plan"]["question"] == question
    assert meta["resolution_used"]


def test_first_turn_all_modes_reach_identical_v5_request(monkeypatch):
    received = []
    monkeypatch.setattr(AdaptiveWorkflow, "_understand", lambda self, state: received.append(deepcopy(state["request"])) or {"decision": "retrieve"})
    for mode in ("none", "history", "structured"):
        operation = MemoryWorkflow(context={}, mode=mode, explicit_fields={"question"}, memory_meta={},
                                   trace_file=None, settings=Settings(_env_file=None))
        operation._understand({"request": AdaptiveRequest(question="三年级数学：长8厘米宽3厘米求周长").model_dump(), "calls": []})
    assert received[0] == received[1] == received[2]


def test_resolution_cannot_invent_numbers_or_recall_evicted_initial_task(monkeypatch):
    operation, _ = memory_operation(monkeypatch, {"standalone_question": "长99厘米宽3厘米求面积", "needs_clarification": False, "clarification": ""})
    update = operation._understand({"request": AdaptiveRequest(question="那面积呢？").model_dump(), "calls": []})
    assert update["decision"] == "clarify"
    operation.context["older_tasks_omitted"] = 2
    update = operation._understand({"request": AdaptiveRequest(question="回到最初那题").model_dump(), "calls": []})
    assert update["decision"] == "clarify" and "窗口" in update["clarification"]


def test_v6_api_round_trip_and_unknown_session(monkeypatch):
    service = ConversationService(workflow_factory=FakeWorkflow)
    monkeypatch.setattr("edu_rag.conversation.service", service)
    with TestClient(app) as client:
        response = client.post("/api/v6/ask", json={"question": "求长方形周长"})
        assert response.status_code == 200
        identity = response.json()["memory"]["session_id"]
        assert client.delete("/api/v6/sessions/"+identity).json() == {"status": "cleared"}
        assert client.post("/api/v6/ask", json={"question": "继续", "session_id": identity}).status_code == 404


def test_memory_resolution_shares_the_v5_http_budget(monkeypatch):
    import json
    from edu_rag.llm import complete
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "standalone_question": "长方形长8厘米宽3厘米，面积是多少？", "needs_clarification": False, "clarification": ""})}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})
    def plan(request, **kwargs):
        complete([{"role": "user", "content": "parse"}], settings=kwargs["settings"], client=kwargs["client"])
    monkeypatch.setattr("edu_rag.rag.construct_plan", plan)
    client = httpx.Client(transport=httpx.MockTransport(respond))
    operation = MemoryWorkflow(context={"tasks": [{"question": "长方形长8厘米宽3厘米求周长"}]}, mode="structured",
                               explicit_fields={"question", "max_model_calls"}, memory_meta={}, trace_file=None,
                               settings=Settings(_env_file=None, deepseek_api_key="fake"), client=client)
    try:
        result = operation.run(AdaptiveRequest(question="那面积呢？", max_model_calls=1))
        assert result["status"] == "budget_exceeded"
        assert len(requests) == 1 and result["workflow"]["budget"]["model_calls"] == 1
        assert result["workflow"]["budget"]["records"][0]["stage"] == "memory_resolve"
    finally:
        client.close()


def test_preview_reads_but_does_not_commit_user_history():
    FakeWorkflow.seen = []
    service = ConversationService(workflow_factory=FakeWorkflow)
    first = service.run(ConversationRequest(question="先只检索"), preview=True)
    service.run(ConversationRequest(question="第一条问答", session_id=first["memory"]["session_id"]))
    assert FakeWorkflow.seen == [{}, {}]


def test_database_uuid_exercise_ids_are_serializable_in_next_turn():
    import json
    identity = uuid4()
    class ExerciseWorkflow(FakeWorkflow):
        def run(self, request, **kwargs):
            result = super().run(request, **kwargs)
            result.update(status="exercises_found", exercises=[{"id": identity, "code": "G3-AREA-001", "stem": "求面积"}])
            return result
    for mode in ("none", "history", "structured"):
        service = ConversationService(workflow_factory=ExerciseWorkflow)
        first = service.run(ConversationRequest(question="给我一道题", memory_mode=mode))
        result = service.run(ConversationRequest(question="再给一道", memory_mode=mode, session_id=first["memory"]["session_id"]))
        assert result["memory"]["turn"] == 2
        snapshot = service.graph.get_state({"configurable": {"thread_id": first["memory"]["session_id"]}}).values
        assert snapshot["turns"][0]["exercises"][0]["id"] == str(identity)
        json.dumps(snapshot["turns"])
