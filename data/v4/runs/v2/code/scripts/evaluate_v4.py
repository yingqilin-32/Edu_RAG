"""Read-only V4 acceptance, frozen labels, resumable original outputs, no label-to-model leakage."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median

from scipy.stats import beta
from sqlalchemy import text

from edu_rag import __version__
from edu_rag.config import PROJECT_ROOT, get_settings
from edu_rag.db import get_engine
from edu_rag.embedding import get_encoder
from edu_rag.workflow import EvidenceWorkflow, WorkflowRequest, WORKFLOW_VERSION
from evaluate_four_way import inventory
from verify_v3 import SnapshotEngine, digest, manifest


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')


def corpus_fingerprint(conn):
    provenance = [dict(row) for row in conn.execute(text(
        'SELECT m.id,m.source_page,m.chunk_index,d.source_uri AS document_uri,e.source_uri AS exercise_uri '
        'FROM public.edu_chunks m LEFT JOIN public.edu_documents d ON d.id=m.document_id '
        'LEFT JOIN public.edu_exercises e ON e.id=m.exercise_id ORDER BY m.id'
    )).mappings()]
    return digest({'base': manifest(conn), 'tags': inventory(conn), 'provenance': provenance})


def measurements(rows, target=0.05):
    negatives = [row for row in rows if not row['answerable']]
    positives = [row for row in rows if row['answerable']]
    answered = {'answered', 'exercises_found', 'evidence_ready'}
    false_answers = sum(row['response']['status'] in answered for row in negatives)
    n = len(negatives)
    def faults(row):
        trace = row['response']['workflow']
        return bool(trace.get('error')) or any(call['status'] not in ('ok', 'fallback') for call in trace['calls'])
    errors = [row['id'] for row in rows if faults(row)]
    rate = false_answers/n if n else None
    positive_rate = sum(row['response']['status'] in answered for row in positives)/len(positives) if positives else None
    upper = float(beta.ppf(0.95, false_answers+1, n-false_answers)) if n and false_answers < n else 1.0
    groups = {}
    for category in sorted({row['category'] for row in rows}):
        items = [row for row in rows if row['category'] == category]
        groups[category] = {'count': len(items), 'statuses': dict(Counter(row['response']['status'] for row in items)),
                            'service_errors': sum(faults(row) for row in items)}
    return {'negatives': n, 'false_answers': false_answers, 'false_answer_rate': rate,
            'one_sided_95_percent_binomial_upper': upper,
            'confidence_note': 'Binomial bound assumes independent representative trials; this authored non-random suite does not establish a production guarantee.',
            'positives': len(positives), 'answerable_answer_rate': positive_rate,
            'service_error_cases': errors, 'status_counts': dict(Counter(row['response']['status'] for row in rows)),
            'parser_fallback_cases': [row['id'] for row in rows if any(call['status'] == 'fallback' for call in row['response']['workflow']['calls'])],
            'retry_distribution': dict(Counter(row['response']['workflow']['retry_count'] for row in rows)),
            'retry_bound_respected': all(row['response']['workflow']['retry_count'] <= 2 and row['response']['workflow']['retrieval_count'] <= 3 for row in rows),
            'citation_checks_passed': all(row['response']['citations'] and
                all(source['id'] in {s['id'] for s in row['response']['sources']} for source in row['response']['citations'])
                for row in rows if row['response']['status'] in answered),
            'median_elapsed_ms': median(row['response']['workflow']['elapsed_ms'] for row in rows) if rows else None,
            'model_calls': sum(len(row['response']['workflow']['calls']) for row in rows),
            'usage': {key: sum(row['response']['usage'].get(key, 0) for row in rows)
                      for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
            'empirical_target_met': bool(n == 60 and len(positives) == 20 and rate <= target and positive_rate >= 0.8 and not errors),
            'category_results': groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-id', default='v1')
    parser.add_argument('--suite', default='eval.json')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_-]{1,40}', args.run_id):
        parser.error('run-id must contain lowercase letters, numbers, hyphens or underscores')
    directory = PROJECT_ROOT / 'data/v4'
    if not re.fullmatch(r'[a-z0-9_-]+\.json', args.suite):
        parser.error('suite must be a JSON filename inside data/v4')
    suite_path = directory / args.suite
    suite = json.loads(suite_path.read_text(encoding='utf-8'))
    cases = suite['cases']
    assert len(cases) == 80 and sum(not case['answerable'] for case in cases) == 60
    assert len({case['id'] for case in cases}) == 80
    assert len({case['question'] for case in cases}) == 80
    settings, engine = get_settings(), get_engine()
    code_paths = ['src/edu_rag/'+name+'.py' for name in ('workflow', 'query', 'rag', 'retrieval', 'llm', 'scope', 'knowledge', 'embedding')]
    code_paths += ['scripts/evaluate_v4.py']
    signature = {'suite_sha256': sha256(suite_path.read_bytes()).hexdigest(), 'workflow': WORKFLOW_VERSION,
                 'model': settings.deepseek_model, 'query_mode': 'deepseek', 'mode': 'hybrid',
                 'code': {path: sha256((PROJECT_ROOT/path).read_bytes()).hexdigest() for path in code_paths}}
    output_dir = directory / 'runs' / args.run_id
    rows = []
    with engine.connect().execution_options(isolation_level='REPEATABLE READ') as conn:
        with conn.begin():
            conn.execute(text('SET TRANSACTION READ ONLY'))
            assert conn.scalar(text('SHOW transaction_read_only')) == 'on'
            signature['corpus_sha256'] = corpus_fingerprint(conn)
            bank = inventory(conn)
            # Validate all four positive exercise labels against actual stored tags before any model call.
            for code, grade, kind, count, difficulty in [('rectangle-area', 3, 'calculation', 2, 2),
                    ('square-area', 3, 'calculation', 2, None), ('money-units', 1, 'calculation', 2, None),
                    ('division-with-remainder', 3, 'calculation', 2, None)]:
                matches = [row for row in bank if row['exercise_id'] and row['grade'] == grade and row['topic_code'] == code
                           and row['question_type'] == kind and (difficulty is None or row['difficulty'] == difficulty)]
                assert len(matches) >= count, f'Invalid positive label: {code}/{kind}/{difficulty}'
            print(f'Preflight: {len(bank)} chunks, 60 negatives + 20 positives, positive exercise labels verified', flush=True)
            if args.preflight_only:
                print(json.dumps(signature, ensure_ascii=False), flush=True)
                return
            output_dir.mkdir(parents=True, exist_ok=True)
            sig_path = output_dir / 'signature.json'
            ledger_path = output_dir / 'results.jsonl'
            if sig_path.exists():
                assert args.resume, 'Run already exists; use --resume for identical configuration or a new run-id.'
                assert json.loads(sig_path.read_text(encoding='utf-8')) == signature, 'Frozen signature changed; preserve this run and use a new run-id.'
                if ledger_path.exists():
                    rows = [json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines() if line]
            else:
                write_json(sig_path, signature)
                write_json(output_dir/'suite.json', suite)
                for relative in code_paths:
                    snapshot = output_dir/'code'/relative
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    snapshot.write_bytes((PROJECT_ROOT/relative).read_bytes())
            done = {row['id'] for row in rows}
            encoder = get_encoder()
            encoder.encode(['V4预热，不计入验收时延'])
            workflow = EvidenceWorkflow(engine=SnapshotEngine(conn), encoder=encoder)
            for case in cases:
                if case['id'] in done:
                    continue
                # Gold answerability/category/rationale are never sent to any model or graph node.
                response = workflow.run(WorkflowRequest(question=case['question'], query_mode='deepseek',
                                                        retrieval_mode='hybrid', max_retries=2))
                row = {**case, 'response': response}
                with ledger_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str)+'\n')
                rows.append(row)
                write_json(output_dir/'progress.json', {'completed': len(rows), 'total': 80, 'metrics': measurements(rows)})
                print(json.dumps({'completed': len(rows), 'id': case['id'], 'answerable': case['answerable'],
                                  'status': response['status'], 'retry_count': response['workflow']['retry_count'],
                                  'error': response['workflow']['error'], 'calls': len(response['workflow']['calls'])}, ensure_ascii=False), flush=True)
    with engine.connect() as conn:
        after = corpus_fingerprint(conn)
    report = {'version': __version__, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'signature': signature, 'suite': suite['set'], 'note': suite['note'],
              'definition': 'False answer = answered/exercises_found/evidence_ready on a pre-labelled unanswerable question. Clarifications and fixed refusals contain no substantive unsupported answer. Availability on positive questions is reported separately, not full answer accuracy.',
              'baseline_note': 'V3 retrieval empty-result tests are not a measured answer-error baseline and are not compared as such.',
              'target': suite['target_false_answer_rate'], 'positive_availability_floor': 0.8,
              'metrics': measurements(rows), 'corpus_unchanged': after == signature['corpus_sha256'],
              'corpus_after': after, 'results': rows}
    write_json(output_dir/'report.json', report)
    print(json.dumps({'report': str(output_dir/'report.json'), 'metrics': report['metrics'], 'corpus_unchanged': report['corpus_unchanged']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
