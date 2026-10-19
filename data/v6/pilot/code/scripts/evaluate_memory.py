"""Paired live multi-turn Memory ablation; same V5 retrieval, gates and budgets."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from edu_rag.assessment import digest
from edu_rag.config import PROJECT_ROOT, get_settings
from edu_rag.conversation import ConversationRequest, ConversationService, MemoryWorkflow, MODES, SUCCESS
from edu_rag.db import get_engine
from edu_rag.embedding import get_encoder
from evaluate_rerank_gate import MemoClient
from evaluate_v4 import corpus_fingerprint, write_json
from verify_v3 import SnapshotEngine

DIRECTORY = PROJECT_ROOT/"data/v6/runs/v1"


def assess(turn, response):
    question = response["resolved_question"]
    source_grades = {s["grade"] for s in response["sources"]}
    checks = {"required_conditions": all(term in question for term in turn.get("must_contain", [])) and
                                      not any(term in question for term in turn.get("must_not_contain", [])),
              "source_scope": bool(source_grades) and source_grades == {turn.get("grade")}}
    if turn.get("semester"):
        checks["source_scope"] &= all(s["semester"] == turn["semester"] for s in response["sources"])
    exercise = turn.get("exercise")
    if exercise:
        checks["exercise_tags"] = response["status"] == "exercises_found" and len(response["exercises"]) == exercise["limit"] and all(
            s.get("knowledge_point_code") == exercise["topic"] and s.get("difficulty") in exercise["difficulty"] and
            s.get("question_type") == exercise["question_type"] for s in response["sources"])
    return {"checks": checks, "answered": response["status"] in SUCCESS,
            "routing_and_conditions_pass": bool(turn.get("answerable") and response["status"] in SUCCESS and all(checks.values())),
            "negative_status_flag": not turn.get("answerable", True) and response["status"] in SUCCESS}


def summarize(rows):
    groups = {}
    for mode in MODES:
        targets = [r for r in rows if r["mode"] == mode and "answerable" in r["turn"]]
        positives = [r for r in targets if r["turn"]["answerable"]]
        negatives = [r for r in targets if not r["turn"]["answerable"]]
        all_turns = [r for r in rows if r["mode"] == mode]
        def avg(values): return mean(values) if values else None
        groups[mode] = {"all_turns": len(all_turns), "positive_followups": len(positives), "negative_followups": len(negatives),
                        "positive_answered": sum(r["assessment"]["answered"] for r in positives),
                        "condition_retention": sum(r["assessment"]["checks"]["required_conditions"] for r in positives),
                        "routing_and_conditions_pass": sum(r["assessment"]["routing_and_conditions_pass"] for r in positives),
                        "negative_status_flags": sum(r["assessment"]["negative_status_flag"] for r in negatives),
                        "model_calls_followups": sum(r["response"]["workflow"]["budget"]["model_calls"] for r in targets),
                        "model_calls_all_turns": sum(r["response"]["workflow"]["budget"]["model_calls"] for r in all_turns),
                        "tokens_followups": sum(r["response"]["usage"]["total_tokens"] for r in targets),
                        "tokens_all_turns": sum(r["response"]["usage"]["total_tokens"] for r in all_turns),
                        "avg_context_characters_followups": avg([r["response"]["memory"]["context_characters"] for r in targets]),
                        "avg_input_tokens_followups": avg([r["response"]["usage"]["prompt_tokens"] for r in targets]),
                        "median_api_latency_proxy_ms": median(r["api_latency_proxy_ms"] for r in targets) if targets else None,
                        "service_errors": sum(bool(r["response"]["workflow"].get("error")) for r in all_turns),
                        "resolution_calls": sum(sum(c["stage"] == "memory_resolve" for c in r["response"]["workflow"]["budget"]["records"]) for r in targets),
                        "status_counts_followups": dict(Counter(r["response"]["status"] for r in targets))}
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DIRECTORY)
    parser.add_argument("--dialog", action="append")
    args = parser.parse_args()
    source = json.loads((PROJECT_ROOT/"data/v6/suite.json").read_text(encoding="utf-8"))
    if args.dialog:
        source["dialogs"] = [d for d in source["dialogs"] if d["id"] in args.dialog]
    settings = get_settings()
    args.output.mkdir(parents=True, exist_ok=True)
    code = {p: (PROJECT_ROOT/p).read_text(encoding="utf-8") for p in ("src/edu_rag/conversation.py", "scripts/evaluate_memory.py", "src/edu_rag/adaptive.py", "src/edu_rag/gating.py", "src/edu_rag/workflow.py", "src/edu_rag/query.py", "src/edu_rag/rag.py", "src/edu_rag/budget.py")}
    signature = {"suite_sha256": digest(source), "code_sha256": digest(code), "model": settings.deepseek_model,
                 "budgets": {"max_retries": 2, "max_branches": 3, "max_model_calls": 10, "max_total_tokens": 30000},
                 "started_at": datetime.now(timezone.utc).isoformat()}
    path = args.output/"signature.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if any(old[k] != signature[k] for k in ("suite_sha256", "code_sha256", "model", "budgets")):
            raise ValueError("Inputs or code changed; choose a new output directory.")
    write_json(path, signature)
    write_json(args.output/"suite.json", source)
    for name, contents in code.items():
        target = args.output/"code"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")
    engine = get_engine()
    client = MemoClient(args.output/"provider_cache", settings)
    rows = []
    try:
        with engine.connect() as connection:
            connection.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            before = corpus_fingerprint(connection)
            factory = partial(MemoryWorkflow, engine=SnapshotEngine(connection), encoder=get_encoder(), settings=settings,
                              client=client, trace_file=None)
            for dialog in source["dialogs"]:
                for mode in MODES:
                    service = ConversationService(workflow_factory=factory)
                    identity = None
                    for index, turn in enumerate(dialog["turns"], 1):
                        start, requests_before = perf_counter(), len(client.served)
                        request = ConversationRequest(question=turn["question"], session_id=identity, memory_mode=mode)
                        response = service.run(request)
                        identity = response["memory"]["session_id"]
                        requests = client.served[requests_before:]
                        row = {"dialog_id": dialog["id"], "category": dialog["category"], "mode": mode, "turn_index": index,
                               "turn": turn, "response": response, "assessment": assess(turn, response) if "answerable" in turn else None,
                               "wall_clock_ms_cached_run": (perf_counter()-start)*1000,
                               "api_latency_proxy_ms": sum(r["provider_elapsed_ms"] for r in requests),
                               "provider_requests": requests}
                        rows.append(row)
                        with (args.output/"results.jsonl").open("w", encoding="utf-8") as stream:
                            for item in rows:
                                stream.write(json.dumps(item, ensure_ascii=False, default=str)+"\n")
                        print(f"{dialog['id']} {mode} turn={index} {response['status']} calls={response['workflow']['budget']['model_calls']}", flush=True)
                    service.delete(identity)
            after = corpus_fingerprint(connection)
        report = {"complete": True, "corpus_sha256_before": before, "corpus_sha256_after": after, "corpus_unchanged": before == after,
                  "modes": summarize(rows), "actual_new_http_calls": sum(not r["cache_hit"] for r in client.served),
                  "actual_new_tokens": sum(r["usage"].get("total_tokens", 0) for r in client.served if not r["cache_hit"]),
                  "limits": ["Authored development dialogs, one response per distinct prompt; not a representative holdout.",
                             "Exact-prompt provider responses shared across arms; outcomes are correlated.",
                             "Routing/condition pass does not establish mathematical answer correctness; see separate Ragas evaluation and answer audit.",
                             "Latency proxy sums original provider times, not end-to-end latency; cached-run wall time is not production latency.",
                             "Structured memory retains four recent records; full history retains at most twelve turns. No LLM narrative summarization.",
                             "In-process checkpoints, no authenticated multi-user hosting or restart persistence."]}
        write_json(args.output/"report.json", report)
        write_json(args.output/"requests.json", client.served)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        client.client.close()


if __name__ == "__main__":
    main()
