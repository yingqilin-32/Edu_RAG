"""Read-only integration checks for the deployed evidence-gate policy and retry bound."""

from datetime import datetime, timezone
from hashlib import sha256
import json

from edu_rag.config import PROJECT_ROOT
from edu_rag.db import get_engine
from edu_rag.embedding import get_encoder
from edu_rag.gating import GatedWorkflow
from edu_rag.deployment import DEFAULT_CONTEXT_POLICY
from edu_rag.workflow import WorkflowRequest
from evaluate_v4 import corpus_fingerprint, write_json


def main():
    cases = json.loads((PROJECT_ROOT/"data/v4/rerank_ablation/v2/suite.json").read_text(encoding="utf-8"))["cases"]
    ids = ("pos-09", "pos-18", "pos-19", "neg-missing_input-04", "neg-provenance-02",
           "neg-partial_requirements-01", "neg-adversarial_or_false_premise-06", "fresh-10")
    engine, encoder = get_engine(), get_encoder()
    with engine.connect() as conn: before = corpus_fingerprint(conn)
    encoder.encode(["warmup"])
    service = GatedWorkflow(context_policy=DEFAULT_CONTEXT_POLICY, engine=engine, encoder=encoder, trace_file=None)
    rows = []
    for case_id in ids:
        case = next(c for c in cases if c["id"] == case_id)
        response = service.run(WorkflowRequest(question=case["question"], query_mode="deepseek", max_retries=2))
        rows.append({"id": case_id, "question": case["question"], "expected_answerable": case["answerable"], "response": response})
        print(json.dumps({"id": case_id, "status": response["status"], "retries": response["workflow"]["retry_count"],
                          "error": response["workflow"]["error"]}), flush=True)
    with engine.connect() as conn: after = corpus_fingerprint(conn)
    checks = {
        "corpus_unchanged": before == after,
        "service_errors_absent": all(not r["response"]["workflow"]["error"] for r in rows),
        "retry_bounds_respected": all(r["response"]["workflow"]["retry_count"] <= 2 and r["response"]["workflow"]["retrieval_count"] <= 3 for r in rows),
        "search_scope_immutable": all(all(s["effective_scope"] == r["response"]["query_plan"]["effective_scope"] and s["filters"] == r["response"]["query_plan"]["filters"] for s in r["response"]["workflow"]["searches"]) for r in rows),
        "negative_cases_never_answered": all(r["response"]["status"] in ("refused", "needs_clarification") for r in rows if not r["expected_answerable"]),
        "positive_cases_completed": all(r["response"]["status"] in ("answered", "exercises_found") for r in rows if r["expected_answerable"]),
        "missing_width_clarifies": next(r for r in rows if r["id"] == "neg-missing_input-04")["response"]["status"] == "needs_clarification",
    }
    report = {"verified_at": datetime.now(timezone.utc).isoformat(), "policy": service.context_policy,
              "code_sha256": sha256((PROJECT_ROOT/"src/edu_rag/gating.py").read_bytes()).hexdigest(),
              "corpus_sha256": before, "checks": checks, "results": rows,
              "note": "Small integration regression with fresh planning and max_retries=2; not an additional independent false-answer-rate estimate."}
    output = PROJECT_ROOT/"data/v4/rerank_ablation/live_integration.json"
    write_json(output, report)
    print(json.dumps({"checks": checks, "output": str(output)}, ensure_ascii=False, indent=2))
    assert all(checks.values())


if __name__ == "__main__": main()
