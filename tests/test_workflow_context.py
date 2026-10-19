from copy import deepcopy

from edu_rag import workflow
from test_workflow import check, judgement, plan, retrieval, runner, source


def test_trimmed_candidate_pool_is_reassessed_before_exercise_display(monkeypatch):
    ready_plan = plan('exercise_search')
    ready_plan['effective_top_k'] = 1
    sources = [source('C1'), source('C2')]
    for s in sources:
        s.update(exercise_id='exercise-'+s['citation'], document_id=None)
    pool = retrieval(sources=sources)
    pool['exercises'] = [{'citation': s['citation'], 'stem': '题干', 'answer': '答案'} for s in sources]
    grades = judgement()
    grades['scores'].append({**grades['scores'][0], 'citation': 'C2'})
    grades['requirements'][0]['citations'] = ['C1', 'C2']
    review = {key: deepcopy(value) for key, value in judgement().items() if key != 'scores'}
    service, queue, requests, client = runner(monkeypatch, [grades, review], ready_plan=ready_plan, result=pool)
    with client:
        response = service.run(workflow.WorkflowRequest(question='给我一道长方形面积题', query_mode='rules'))
    assert response['status'] == 'exercises_found'
    assert len(response['exercises']) == 1
    assert [call['stage'] for call in response['workflow']['calls']] == ['rerank', 'evaluate']
    assert not queue


def test_trimmed_context_cannot_inherit_unsupported_full_pool_coverage(monkeypatch):
    ready_plan = plan()
    ready_plan['effective_top_k'] = 1
    pool = retrieval(sources=[source('C1'), source('C2')])
    grades = judgement()
    grades['scores'].append({**grades['scores'][0], 'citation': 'C2'})
    review = {key: deepcopy(value) for key, value in judgement(answerable=False).items() if key != 'scores'}
    service, queue, requests, client = runner(monkeypatch, [grades, review], ready_plan=ready_plan, result=pool)
    with client:
        response = service.run(workflow.WorkflowRequest(question='需要多项依据的问题', query_mode='rules', max_retries=0))
    assert response['status'] == 'refused' and not queue
