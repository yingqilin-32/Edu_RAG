import pytest

from edu_rag import query, workflow
from test_workflow import judgement, runner


def test_elapsed_time_is_not_filtered_to_reading_clock_face():
    topics = [{'id': 'clock', 'code': 'reading-clocks', 'title': '钟面和时刻'},
              {'id': 'duration', 'code': 'time-duration', 'title': '时分与经过时间'}]
    assert query.resolve_topics('下午2时20分开始，40分钟后结束时刻是多少？', topics) == [
        {'id': 'duration', 'title': '时分与经过时间'}]
    assert query.resolve_topics('40分钟后结束时刻是多少？', topics[:1]) == []


@pytest.mark.parametrize('question', ['提供三年级难度4的长方形面积计算题。', '请出2道长方形面积练习题', '搜索三年级长方形面积计算题'])
def test_explicit_exercise_verbs_preserve_lookup_intent(question):
    draft = query.rule_draft(question)
    assert draft.intent == 'exercise_search'
    if '难度4' in question:
        assert draft.filters.difficulty == 4


def test_absent_optional_clarification_does_not_discard_valid_evidence(monkeypatch):
    grades = judgement()
    grades.pop('clarification')
    service, queue, requests, client = runner(monkeypatch, [grades])
    with client:
        response = service.run(workflow.WorkflowRequest(question='解释长方形面积公式', query_mode='rules'), preview=True)
    assert response['status'] == 'evidence_ready' and not response['workflow']['error']


def test_empty_requirements_and_null_quote_cannot_enable_an_answer(monkeypatch):
    grades = {**judgement(), 'requirements': [], 'clarification': None}
    grades['scores'][0]['quote'] = None
    service, queue, requests, client = runner(monkeypatch, [grades])
    with client:
        response = service.run(workflow.WorkflowRequest(question='解释未收录的知识', query_mode='rules', max_retries=0))
    assert response['status'] == 'refused' and not response['workflow']['error']
    assert response['workflow']['assessment']['requirement_coverage'] == 0


def test_rewrite_may_echo_existing_fact_but_not_change_its_unit(monkeypatch):
    service, queue, requests, client = runner(monkeypatch, [judgement(answerable=False), {'additional_terms': ['长8厘米面积']}, judgement()])
    with client:
        response = service.run(workflow.WorkflowRequest(question='长方形长8厘米面积怎么算', query_mode='rules'), preview=True)
    assert response['status'] == 'evidence_ready' and response['workflow']['retry_count'] == 1
    service, queue, requests, client = runner(monkeypatch, [judgement(answerable=False), {'additional_terms': ['长8米面积']}])
    with client:
        response = service.run(workflow.WorkflowRequest(question='长方形长8厘米面积怎么算', query_mode='rules'), preview=True)
    assert response['status'] == 'refused' and response['workflow']['retrieval_count'] == 1
