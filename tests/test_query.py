import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from edu_rag import rag
from edu_rag.config import Settings
from edu_rag.main import app
from edu_rag.query import QueryDraft, build_plan, rule_draft, resolve_topics
from edu_rag.scope import CurriculumScope

COURSES = [dict(id="10000000-0000-4000-8000-000000000001", code="G3", grade=3, semester="full_year", subject="math", textbook_edition="自编演示资料"),
           dict(id="e311f354-e970-585d-835b-5fa0b9a9a21f", code="G1F", grade=1, semester="first", subject="math", textbook_edition="一年级原创补充（参考目录，非教材）"),
           dict(id="0c277934-4ac3-5a66-9a22-32df730d8e1c", code="G1S", grade=1, semester="second", subject="math", textbook_edition="一年级原创补充（参考目录，非教材）")]


@pytest.mark.parametrize("question,intent,summary_scope", [
    ("总结一年级下学期数学有哪些知识点", "knowledge_summary", "overview"),
    ("梳理长方形周长和面积的公式", "knowledge_summary", "focused"),
    ("找3道一年级下册人民币练习题", "exercise_search", "focused"),
    ("长8厘米宽3厘米的长方形周长是多少", "problem_solving", "focused"),
    ("什么是单位分数", "knowledge_qa", "focused"),
    ("总结这道题9+6的解法", "problem_solving", "focused"),
])
def test_rule_intents(question, intent, summary_scope):
    draft = rule_draft(question, COURSES)
    assert draft.intent == intent and draft.summary_scope == summary_scope


def plan(question, **kwargs):
    return build_plan(question, kwargs.pop("scope", CurriculumScope()), mode=kwargs.pop("mode", "rules"), courses=COURSES, **kwargs)


def test_metadata_and_requested_limit_are_bound_to_the_plan():
    result = plan("找3道一年级下册数学人民币练习题")
    assert result["status"] == "ready" and result["content_kind"] == "exercise"
    assert result["effective_scope"]["grade"] == 1 and result["effective_scope"]["semester"] == "second"
    assert result["effective_top_k"] == 3


@pytest.mark.parametrize("question", ["一年级和三年级都总结一下", "找11道题", "找十五道题", "总结十五年级数学", "讲解这道题", "总结这个知识点"])
def test_ambiguous_or_out_of_bound_requests_require_clarification(question):
    assert plan(question)["status"] == "needs_clarification"


@pytest.mark.parametrize("scope", [CurriculumScope(grade=3), CurriculumScope(course_id=COURSES[0]["id"])])
def test_natural_conditions_cannot_override_explicit_scope(scope):
    assert plan("总结一年级数学知识点", scope=scope)["status"] == "needs_clarification"


def test_unknown_textbook_is_not_mapped_to_authored_material():
    assert plan("总结一年级人教版数学知识点")["status"] == "no_matching_scope"
    assert plan("总结一年级青岛版数学知识点")["status"] == "no_matching_scope"


def test_normalization_keeps_the_exact_stored_textbook_label():
    result = plan("总结一年级下学期一年级原创补充（参考目录，非教材）的数学知识点")
    assert result["status"] == "ready"
    assert result["extracted_filters"]["textbook_edition"] == COURSES[2]["textbook_edition"]


def model_response(payload, check=None, status=200):
    def respond(request):
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert "json" in body["messages"][0]["content"].lower()
        if check:
            check(body)
        return httpx.Response(status, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload, ensure_ascii=False)}}],
                                           "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})
    return httpx.Client(transport=httpx.MockTransport(respond))


def test_llm_json_plan_preserves_numbers_units_and_literal_filters():
    draft = QueryDraft(intent="exercise", semantic_query="长8米宽5米周长？", keywords=["长方形", "99"],
                       filters={}).model_dump()
    with model_response(draft) as client:
        result = plan("一年级下学期长8厘米宽3厘米，周长是多少？", mode="deepseek",
                      settings=Settings(_env_file=None, deepseek_api_key="test"), client=client)
    assert result["origin"] == "deepseek" and result["status"] == "ready"
    assert result["semantic_query"] == "一年级下学期长8厘米宽3厘米，周长是多少？"
    assert "99" not in result["bm25_query"]
    assert result["effective_scope"]["grade"] == 1 and result["effective_scope"]["semester"] == "second"
    assert result["usage"]["total_tokens"] == 150 and result["warnings"]


@pytest.mark.parametrize("payload,status", [({"sql": "DROP TABLE edu_chunks"}, 200), ({}, 402)])
def test_invalid_or_failed_llm_falls_back_to_rules_without_relaxing_filters(payload, status):
    with model_response(payload, status=status) as client:
        result = plan("查找一年级下学期人民币题目", mode="deepseek", scope=CurriculumScope(grade=1),
                      settings=Settings(_env_file=None, deepseek_api_key="test"), client=client)
    assert result["origin"] == "rules_fallback" and result["warnings"]
    assert result["effective_scope"]["grade"] == 1 and result["effective_scope"]["semester"] == "second"
    assert "sql" not in result


def test_invented_exercise_id_is_not_executed():
    draft = QueryDraft(intent="exercise", semantic_query="找人民币练习", exercise_code="V1-FAKE-001").model_dump()
    with model_response(draft) as client:
        result = plan("找人民币练习", mode="deepseek", settings=Settings(_env_file=None, deepseek_api_key="test"), client=client)
    assert result["status"] == "needs_clarification"


def test_exact_code_is_copied_and_is_the_only_lexical_query():
    result = plan("请查题号 v1-time-duration-002 的题目")
    assert result["exercise_code"] == result["bm25_query"] == "V1-TIME-DURATION-002"


def test_empty_explicit_scope_does_not_pay_for_planning():
    def forbidden(request):
        pytest.fail("An empty explicit scope must not call DeepSeek")
    with httpx.Client(transport=httpx.MockTransport(forbidden)) as client:
        result = plan("总结数学", scope=CurriculumScope(grade=12), mode="deepseek", client=client)
    assert result["status"] == "no_matching_scope"


def test_overview_uses_catalog_coverage_and_skips_embedding(monkeypatch):
    rows = [dict(id="a", content="知识一", knowledge_point_id="kp1", exercise_id=None, exercise_answer=None, exercise_explanation=None),
            dict(id="b", content="知识一第二段", knowledge_point_id="kp1", exercise_id=None, exercise_answer=None, exercise_explanation=None),
            dict(id="c", content="知识二", knowledge_point_id="kp2", exercise_id=None, exercise_answer=None, exercise_explanation=None)]
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value.all.return_value = rows
    monkeypatch.setattr(rag, "construct_plan", lambda *a, **k: plan("总结一年级数学知识点"))
    monkeypatch.setattr(rag.knowledge, "knowledge_points", lambda *a, **k: [dict(id=f"kp{i}", title=f"知识{i}") for i in (1,2,3)])
    encoder = SimpleNamespace(model_id="test", encode=MagicMock(side_effect=AssertionError("Overview needs no query vector")))
    result = rag.search(rag.SearchRequest(question="总结一年级数学知识点", query_mode="rules", summary_max_sources=1), engine=engine, encoder=encoder)
    assert result["retrieval"]["mode"] == "catalog"
    coverage = result["query_plan"]["coverage"]
    assert coverage["catalog_topics"] == 3 and coverage["eligible_topics"] == 2 and coverage["selected_topics"] == 1
    assert coverage["partial"] and coverage["omitted_topics"] == ["知识2", "知识3"]
    assert result["sources"][0]["citation"] == "C1"


def test_blocked_plan_skips_embedding_retrieval_and_answer(monkeypatch):
    blocked = plan("总结一年级数学知识点", scope=CurriculumScope(grade=3))
    monkeypatch.setattr(rag, "construct_plan", lambda *a, **k: blocked)
    monkeypatch.setattr(rag, "get_engine", lambda: object())
    monkeypatch.setattr(rag, "get_encoder", lambda: pytest.fail("Blocked query loaded encoder"))
    monkeypatch.setattr(rag, "generate", lambda *a, **k: pytest.fail("Blocked query called answer model"))
    result = rag.ask(rag.SearchRequest(question="总结一年级数学知识点", query_mode="rules"))
    assert result["status"] == "needs_clarification" and result["sources"] == [] and result["model"] is None


def test_summary_prompt_includes_coverage_without_changing_original_question():
    def check(body):
        # generation uses regular output, unlike the planner
        assert "coverage" in body["messages"][0]["content"]
        assert json.loads(body["messages"][1]["content"])["question"] == "总结数学"
    def respond(request):
        body = json.loads(request.content)
        check(body)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "仅总结已提供部分。[C1]"}}]})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = rag.generate("总结数学", [dict(citation="C1", source_title="知识卡", content="知识")],
                              query_plan={"intent": "summary", "coverage": {"partial": True}},
                              settings=Settings(_env_file=None, deepseek_api_key="test"), client=client)
    assert result["status"] == "answered"


def test_new_plan_endpoint_validates_request_without_inference(monkeypatch):
    monkeypatch.setattr(rag, "construct_plan", lambda request: {"intent": "summary", "origin": request.query_mode})
    with TestClient(app) as client:
        assert client.post("/api/v3/plan", json={"question": "总结数学", "query_mode": "rules"}).status_code == 200
        assert client.post("/api/v3/search", json={"question": "总结数学", "query_mode": "unknown"}).status_code == 422


def test_literal_concepts_resolve_only_to_matching_stored_topics():
    topics = [dict(id="a", title="长方形周长"), dict(id="b", title="长方形面积"),
              dict(id="c", title="正方形面积"), dict(id="d", title="人民币单位及换算")]
    assert [t["id"] for t in resolve_topics("长方形的周长和面积", topics)] == ["a", "b"]
    assert resolve_topics("找三道人民币题", topics) == [dict(id="d", title="人民币单位及换算")]
    assert resolve_topics("请总结凑十方法", topics) == []
    assert resolve_topics("53个人坐船，每条最多8人", topics) is None
