"""Offline paired comparisons and explicit audit ledger; never calls an LLM."""

from collections import Counter
from hashlib import sha256
import json
from statistics import median

from evaluate_rerank_gate import DIRECTORY, ARMS, SUCCESS
from evaluate_v4 import write_json


def main():
    rows = [json.loads(line) for line in (DIRECTORY/"results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 460
    indexed = {(r["id"], r["arm"]): r for r in rows}
    comparisons = {}
    for arm in ARMS[1:]:
        pairs = [(r, indexed[(r["id"], arm)]) for r in rows if r["arm"] == "retrieval_order"]
        comparisons[arm] = {
            "positive_completions_gained": [a["id"] for a, b in pairs if a["answerable"] and a["response"]["status"] not in SUCCESS and b["response"]["status"] in SUCCESS],
            "positive_completions_lost": [a["id"] for a, b in pairs if a["answerable"] and a["response"]["status"] in SUCCESS and b["response"]["status"] not in SUCCESS],
            "context_order_changed": sum(a["selected_ids"] != b["selected_ids"] for a, b in pairs),
            "context_members_changed": sum(set(a["selected_ids"]) != set(b["selected_ids"]) for a, b in pairs),
            "nominal_extra_calls": sum(len(b["response"]["workflow"]["calls"])-len(a["response"]["workflow"]["calls"]) for a, b in pairs),
            "nominal_extra_tokens": sum(b["response"]["usage"]["total_tokens"]-a["response"]["usage"]["total_tokens"] for a, b in pairs),
        }
    negative_status_flags = [{"id": r["id"], "arm": r["arm"], "question": r["question"], "answer": r["response"]["answer"]}
                             for r in rows if not r["answerable"] and r["response"]["status"] in SUCCESS]
    for flag in negative_status_flags:
        flag["audit"] = {
            "neg-adversarial_or_false_premise-05": "The text correctly refuses to invent a bank balance. It is marked answered, so counted by the frozen status-based metric, but it is NOT an unsupported factual answer.",
            "neg-adversarial_or_false_premise-06": "Returns real square exercises instead of rejecting a request for an author's private ID. Task-routing failure, not a fabricated ID. Exercise routes bypass answer post-verification, illustrating the pre-generation gate's role."
        }.get(flag["id"], "REQUIRES_AUDIT")
    source_sets = [r for r in rows if r["response"]["status"] in SUCCESS]
    checks = {"cases": len({r["id"] for r in rows}), "arms_per_case": dict(Counter(Counter(r["id"] for r in rows).values())),
              "all_success_citations_selected": all(r["response"]["citations"] and all(s["id"] in {v["id"] for v in r["response"]["sources"]} for s in r["response"]["citations"]) for r in source_sets),
              "all_contexts_within_budget": all(len(r["selected_ids"]) <= r["budget"] for r in rows),
              "rerank_calls_absent_in_control": all(not any(c["stage"] == "rerank" for c in r["response"]["workflow"]["calls"]) for r in rows if r["arm"] == "retrieval_order"),
              "all_first_pass_only": all(r["response"]["workflow"]["retry_count"] == 0 for r in rows)}
    per_partition = {}
    for arm in ARMS:
        per_partition[arm] = {}
        for partition in ("reused_regression", "new_authored_variants"):
            pos = [r for r in rows if r["arm"] == arm and r["partition"] == partition and r["answerable"]]
            per_partition[arm][partition] = {"positive_count": len(pos), "completed": sum(r["response"]["status"] in SUCCESS for r in pos)}
    artifact = {"checks": checks, "comparisons_against_retrieval_order": comparisons,
                "partition_metrics": per_partition, "negative_status_flag_audit": negative_status_flags,
                "inference_limit": "Paired development sample, one provider output per unique prompt. Exact prompt caching controls variation only for identical inputs; small gains/losses may reflect model variation. No claim of rerank equivalence or production guarantee.",
                "metric_limit": "The preregistered negative metric is status based. Report the semantic audit alongside it; never describe all status flags as factual hallucinations."}
    write_json(DIRECTORY/"paired_analysis.json", artifact)
    # Compact unique answers for source-level review by the assistant, not a second LLM judge.
    answers = {}
    for r in rows:
        if not r["answerable"] or r["response"]["status"] not in SUCCESS: continue
        key = sha256((r["id"]+r["response"]["answer"]).encode()).hexdigest()
        if key not in answers:
            answers[key] = {"id": r["id"], "question": r["question"], "expected_answer": r.get("expected_answer"),
                            "answer": r["response"]["answer"], "exercises": r["response"].get("exercises", []), "arms": []}
        answers[key]["arms"].append(r["arm"])
    write_json(DIRECTORY/"answers_for_audit.json", list(answers.values()))
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    print("Unique positive answers to audit:", len(answers))


if __name__ == "__main__": main()
