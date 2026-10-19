"""Offline workflow-path, cost and answer audit for the frozen V5 experiment."""

from collections import Counter
from hashlib import sha256
import json
from statistics import median

from evaluate_v5 import DIRECTORY, POLICIES, SUCCESS
from evaluate_v4 import write_json


def main():
    rows = [json.loads(s) for s in (DIRECTORY/"results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 276
    inputs = json.loads((DIRECTORY/"frozen_inputs.json").read_text(encoding="utf-8"))["cases"]
    index = {(r["id"], r["policy"]): r for r in rows}
    checks = {"three_results_per_case": all(n == 3 for n in Counter(r["id"] for r in rows).values()),
              "model_call_caps": all(r["response"]["workflow"]["budget"]["model_calls"] <= 10 for r in rows),
              "retry_caps": all(r["response"]["workflow"]["retry_count"] <= 2 for r in rows),
              "branch_caps": all(len(r["response"]["workflow"]["branch_history"]) <= 6 and r["response"]["workflow"]["retrieval_count"] <= 7 for r in rows),
              "immutable_scope": all(all(s["effective_scope"] == inputs[r["id"]]["plan"]["effective_scope"] and s["filters"] == inputs[r["id"]]["plan"]["filters"] for s in r["response"]["workflow"]["searches"]) for r in rows),
              "original_question_preserved_in_branches": all(all(r["question"] in s["semantic_query"] for s in r["response"]["workflow"]["searches"] if s.get("branch")) for r in rows),
              "output_citations_from_selected_sources": all(r["response"]["citations"] and all(s["id"] in {v["id"] for v in r["response"]["sources"]} for s in r["response"]["citations"]) for r in rows if r["response"]["status"] in SUCCESS),
              "single_pass_does_not_retry": all(r["response"]["workflow"]["retry_count"] == 0 for r in rows if r["policy"] == "single_pass")}
    comparisons, timings, examples = {}, {}, []
    for policy in POLICIES:
        items = [r for r in rows if r["policy"] == policy]
        partitions = {}
        for partition in ("simple_control", "new_authored_complex", "reused_negative"):
            values = [r for r in items if r["partition"] == partition]
            def model_ms(r): return sum(p["provider_elapsed_ms"] for p in r["provider_requests"])
            def retry_ms(r): return sum(e["elapsed_ms"] for e in r["response"]["workflow"]["events"] if e["node"] == "retrieve_branches")
            partitions[partition] = {"median_model_ms_excluding_common_planner": median(model_ms(r) for r in values),
                                     "median_retry_retrieval_ms": median(retry_ms(r) for r in values),
                                     "median_model_plus_retry_ms": median(model_ms(r)+retry_ms(r) for r in values)}
        timings[policy] = partitions
        if policy == "single_pass": continue
        gained, lost = [], []
        for r in items:
            base = index[(r["id"], "single_pass")]
            if not r["answerable"]: continue
            if base["response"]["status"] not in SUCCESS and r["response"]["status"] in SUCCESS:
                initial = inputs[r["id"]]["result"]
                initial_ids = {str(s["id"]) for s in initial["sources"]} if initial else set()
                new_ids = [s["id"] for s in r["response"]["sources"] if str(s["id"]) not in initial_ids]
                gained.append({"id": r["id"], "new_selected_evidence_ids": new_ids,
                               "baseline_reason": base["response"]["workflow"]["reason"],
                               "retry_count": r["response"]["workflow"]["retry_count"]})
            if base["response"]["status"] in SUCCESS and r["response"]["status"] not in SUCCESS: lost.append(r["id"])
        comparisons[policy] = {"gained": gained, "lost": lost}
    for r in rows:
        if r["policy"] == "adaptive" and r["response"]["workflow"]["retry_count"]:
            examples.append({"id": r["id"], "question": r["question"], "status": r["response"]["status"],
                             "stop_reason": r["response"]["workflow"]["stop_reason"],
                             "branch_history": r["response"]["workflow"]["branch_history"],
                             "model_calls": r["response"]["workflow"]["budget"]["model_calls"]})
    unique = {}
    for r in rows:
        if not r["answerable"] or r["response"]["status"] not in SUCCESS: continue
        key = sha256((r["id"]+r["response"]["answer"]).encode()).hexdigest()
        if key not in unique:
            unique[key] = {"id": r["id"], "question": r["question"], "expected_answer": r.get("expected_answer"),
                           "answer": r["response"]["answer"], "policies": []}
        unique[key]["policies"].append(r["policy"])
    write_json(DIRECTORY/"answers_for_audit.json", list(unique.values()))
    analysis = {"checks": checks, "paired_completion_changes": comparisons, "timings": timings,
                "timing_note": "Replay original provider duration for identical cached requests, plus measured retry embedding/database work. Excludes common planning and first retrieval; not HTTP end-to-end latency or concurrency throughput.",
                "unique_answers_for_review": len(unique), "adaptive_retry_paths": examples,
                "negative_status_flags": [{"id": r["id"], "policy": r["policy"], "answer": r["response"]["answer"]} for r in rows if not r["answerable"] and r["response"]["status"] in SUCCESS],
                "positive_stops": [{"id": r["id"], "policy": r["policy"], "status": r["response"]["status"],
                                    "reason": r["response"]["workflow"]["reason"], "stop_reason": r["response"]["workflow"]["stop_reason"]}
                                   for r in rows if r["answerable"] and r["response"]["status"] not in SUCCESS]}
    write_json(DIRECTORY/"analysis.json", analysis)
    print(json.dumps({"checks": checks, "paired_completion_changes": comparisons, "unique_answers": len(unique)}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
