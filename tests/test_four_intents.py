from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from edu_rag import rag
from edu_rag.query import QueryDraft, build_plan, resolve_chapter, rule_draft
from edu_rag.scope import CurriculumScope
from edu_rag.retrieval import select_problem_context

COURSES = [dict(id="10000000-0000-4000-8000-000000000001", code="G3", grade=3, semester="full_year", subject="math", textbook_edition="原创")]


def test_requested_json_contract_and_unknown_fields():
    data = dict(intent="exercise_search", search_text="一元二次方程 计算题", filters={"subject":"math", "grade":9,
                "knowledge_point":"一元二次方程", "question_type":"calculation", "difficulty":"medium"}, limit=5, needs_clarification=False)
    draft = QueryDraft.model_validate(data)
    assert draft.filters.grade == 9 and draft.filters.difficulty == "medium" and draft.limit == 5
    with pytest.raises(ValidationError):
        QueryDraft.model_validate({**data, "sql": "SELECT anything"})


@pytest.mark.parametrize("question,intent,task", [
    ("配方法为什么要两边同时加一个数？", "knowledge_qa", None),
    ("总结初三一元二次方程这一章", "knowledge_summary", None),
    ("给我5道中等难度的一元二次方程计算题", "exercise_search", "find"),
    ("这道方程题怎么解？", "problem_solving", "solve"),
    ("总结这道题9+6的解题方法", "problem_solving", "summarize"),
])
def test_four_intents_and_exercise_summary_subtask(question, intent, task):
    draft = rule_draft(question)
    assert draft.intent == intent and draft.exercise_task == task


@pytest.mark.parametrize("text,levels", [("简单",[1,2]), ("中等难度",[3]), ("困难",[4,5]), ("难度2",[2])])
def test_difficulty_buckets_are_fixed_independently_of_available_data(text, levels):
    plan = build_plan("给我5道三年级"+text+"的长方形面积计算题", CurriculumScope(grade=3), mode="rules", courses=COURSES)
    assert plan["question_type"] == "calculation" and plan["difficulty_levels"] == levels


def test_exercise_search_presents_stored_questions_without_answer_api(monkeypatch):
    plan = {"status":"ready", "intent":"exercise_search", "requested_limit":5}
    result = {"sources":[{"citation":"C1"}], "query_plan":plan,
              "exercises":[{"stem":"4元3角是多少角？", "answer":"43角", "explanation":"4元=40角"}]}
    monkeypatch.setattr(rag, "search", lambda request: result)
    monkeypatch.setattr(rag, "generate", lambda *a,**k: pytest.fail("Exercise lookup must not generate new questions"))
    answer = rag.ask(rag.SearchRequest(question="给我5道人民币题", query_mode="rules"))
    assert answer["status"] == "exercises_found" and answer["model"] is None and len(answer["exercises"]) == 1
    assert "少于请求的 5 道" in answer["answer"]


def test_problem_solving_retrieves_knowledge_and_examples_with_original_stem(monkeypatch):
    question = "长8厘米宽3厘米，长方形的周长是多少？"
    plan = build_plan(question, CurriculumScope(grade=3), mode="rules", courses=COURSES)
    assert plan["intent"] == "problem_solving" and plan["content_kind"] == "all"
    monkeypatch.setattr(rag, "construct_plan", lambda *a,**k: plan)
    rows = [dict(id="a", content="长方形周长=(长+宽)×2。", similarity=.9, document_id="doc", exercise_id=None, exercise_answer=None, exercise_explanation=None),
            dict(id="b", content="长9厘米宽4厘米，周长？", similarity=.8, document_id=None, exercise_id="example", exercise_answer="26厘米", exercise_explanation="周长=(9+4)×2")]
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value.all.return_value = rows
    encoder = SimpleNamespace(model_id="test", encode=MagicMock(return_value=[[1.0]]))
    result = rag.search(rag.SearchRequest(question=question, query_mode="rules"), engine=engine, encoder=encoder)
    assert len(result["sources"]) == 2
    assert encoder.encode.call_args.args[0] == [question]
    assert conn.execute.call_args.args[1]["kind"] == "all"


def test_chapter_resolution_requires_registered_hierarchy_and_stays_in_course():
    topics = [dict(id="root", title="一元二次方程", code="quadratic", parent_id=None, course_id="g9"),
              dict(id="child", title="配方法", code="completing-square", parent_id="root", course_id="g9"),
              dict(id="nested", title="解法", code="methods", parent_id="root", course_id="g9"),
              dict(id="leaf", title="公式法", code="formula", parent_id="nested", course_id="g9"),
              dict(id="foreign", title="其他课程", code="foreign", parent_id="root", course_id="other")]
    assert {t["id"] for t in resolve_chapter("一元二次方程", topics)} == {"child","leaf"}
    assert resolve_chapter("配方法", topics) == []
    assert resolve_chapter("不存在的章节", topics) == []


def test_prepared_plan_cannot_relax_request_scope():
    plan = build_plan("总结数学", CurriculumScope(grade=3), mode="rules", courses=COURSES)
    with pytest.raises(ValueError, match="conflicts"):
        rag.search(rag.SearchRequest(question="总结数学", grade=1, query_mode="rules"), engine=object(), prepared_plan=plan)


def test_missing_problem_and_explicit_chapter_are_parsed_without_guessing():
    draft = rule_draft("这道方程题怎么解？")
    assert draft.needs_clarification
    chapter = rule_draft("总结初三一元二次方程这一章")
    assert chapter.filters.grade == 9 and chapter.filters.chapter == "一元二次方程"


def test_solving_context_includes_an_explanation_without_reordering_rank_scores():
    ranked = [dict(id=str(i), document_id=None, exercise_id=str(i)) for i in range(5)]
    ranked.append(dict(id="knowledge", document_id="doc", exercise_id=None))
    assert [r["id"] for r in select_problem_context(ranked, 3)] == ["0", "1", "knowledge"]
    assert select_problem_context(ranked, 1) == ranked[:1]
