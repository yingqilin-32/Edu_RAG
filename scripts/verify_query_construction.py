"""Read-only query-construction verification; optional five real DeepSeek calls."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import text

from edu_rag import __version__, rag
from edu_rag.config import PROJECT_ROOT
from edu_rag.db import get_engine
from edu_rag.embedding import get_encoder
from edu_rag.main import app
from verify_v3 import SnapshotEngine, digest, manifest


def compact(result):
    fields = ("id", "citation", "course_id", "knowledge_point_id", "knowledge_point_code", "knowledge_point_title",
              "document_id", "exercise_id", "exercise_code", "source_title", "similarity", "dense_rank", "bm25_rank", "rrf_score")
    return {"query_plan": result["query_plan"], "retrieval": result["retrieval"],
            "sources": [{key: source.get(key) for key in fields} for source in result["sources"]]}


def check(case, result):
    plan = result["query_plan"]
    assert plan["status"] == case.get("status", "ready"), (case["id"], plan)
    if "intent" in case:
        assert plan["intent"] == case["intent"], (case["id"], plan)
    if "strategy" in case:
        assert plan["strategy"] == case["strategy"]
    if "sources" in case:
        assert len(result["sources"]) == case["sources"], (case["id"], len(result["sources"]))
    for i, source in enumerate(result["sources"], 1):
        assert source["citation"] == f"C{i}"
        for key, value in case["scope"].items():
            assert str(source[key]) == str(value)
        if plan["intent"] == "summary":
            assert source["document_id"] is not None and source["exercise_id"] is None
        if plan["intent"] == "exercise":
            assert source["exercise_id"] is not None and source["document_id"] is None
        if "topic_codes" in case:
            assert source["knowledge_point_code"] in case["topic_codes"]
        if "exercise_code" in case:
            assert source["exercise_code"] == case["exercise_code"]
    if "topics" in case:
        assert len({str(source["knowledge_point_id"]) for source in result["sources"]}) == case["topics"]
    if "omitted_topics" in case:
        assert plan["coverage"]["partial"] and len(plan["coverage"]["omitted_topics"]) == case["omitted_topics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()
    directory = PROJECT_ROOT / "data/v3/query_construction"
    eval_path = directory / "eval.json"
    suite = json.loads(eval_path.read_text(encoding="utf-8"))
    report = {"version": __version__, "set": suite["set"], "note": suite["note"],
              "evaluation_sha256": sha256(eval_path.read_bytes()).hexdigest(), "rules_cases": [], "llm_cases": [], "answers": [],
              "paid_api_attempts": 0, "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    def save():
        (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
    engine, encoder = get_engine(), get_encoder()
    encoder.encode(["查询构建验收预热"])
    cached = {}
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            assert conn.scalar(text("SHOW transaction_read_only")) == "on"
            before = digest(manifest(conn))
            snapshot_engine = SnapshotEngine(conn)
            with patch("edu_rag.rag.get_engine", return_value=snapshot_engine), patch("edu_rag.rag.get_encoder", return_value=encoder):
                for case in suite["cases"]:
                    request = rag.SearchRequest(question=case["question"], **case["scope"], query_mode="rules",
                                                summary_max_sources=case.get("summary_max_sources", 20))
                    result = rag.search(request, engine=snapshot_engine, encoder=encoder)
                    check(case, result)
                    cached[case["id"]] = result
                    row = {"id": case["id"], "question": case["question"], "passed": True, **compact(result)}
                    if case.get("compare_raw"):
                        raw = rag.search(request.model_copy(update={"query_mode": "off"}), engine=snapshot_engine, encoder=encoder)
                        row["raw_v3_baseline"] = {"sources": len(raw["sources"]),
                            "distinct_topics": len({str(s["knowledge_point_id"]) for s in raw["sources"]}),
                            "knowledge_cards": sum(s["document_id"] is not None for s in raw["sources"]),
                            "exercise_codes": [s["exercise_code"] for s in raw["sources"]]}
                    report["rules_cases"].append(row)
                    print(f"Rules {case['id']}: {result['query_plan']['status']}, {len(result['sources'])} sources", flush=True)
                with patch("edu_rag.rag.generate", side_effect=AssertionError("Blocked or empty query called answer model")):
                    for case in suite["cases"]:
                        if case.get("sources") != 0:
                            continue
                        answer = rag.ask(rag.SearchRequest(question=case["question"], **case["scope"], query_mode="rules"))
                        assert answer["sources"] == [] and answer["model"] is None
                with TestClient(app) as client:
                    response = client.post("/api/v3/plan", json={"question": "总结三年级数学知识点", "grade": 3, "query_mode": "rules"})
                    assert response.status_code == 200 and response.json()["strategy"] == "catalog_overview"
                    response = client.post("/api/v3/search", json={"question": "找一年级下学期人民币练习题", "query_mode": "rules"})
                    assert response.status_code == 200
                    assert all(s["knowledge_point_code"] == "money-units" and s["semester"] == "second" for s in response.json()["sources"])

                # Existing retrieval mode remains reproducible when query construction is off.
                old = json.loads((PROJECT_ROOT / "data/v3/retrieval_report.json").read_text(encoding="utf-8"))
                verified = []
                for case in old["results"]:
                    if case["id"] not in ("g3-01", "g3-10", "g1-second-03", "identifier-02", "identifier-07"):
                        continue
                    for mode in ("dense", "hybrid"):
                        result = rag.search(rag.SearchRequest(question=case["question"], **case["scope"], retrieval_mode=mode), engine=snapshot_engine, encoder=encoder)
                        assert [str(s["id"]) for s in result["sources"]] == [s["id"] for s in case["modes"][mode]["top_5"]]
                        verified.append(dict(id=case["id"], mode=mode, unchanged=True))
                report["previous_v3_regression_checks"] = verified
                report["empty_queries_skip_answer_model"] = True
                save()
                if args.with_llm:
                    for identity in ("summary-g1", "summary-focused", "exercise-list"):
                        case = next(case for case in suite["cases"] if case["id"] == identity)
                        report["paid_api_attempts"] += 1
                        result = rag.search(rag.SearchRequest(question=case["question"], **case["scope"], query_mode="deepseek"), engine=snapshot_engine, encoder=encoder)
                        check(case, result)
                        report["llm_cases"].append({"id": identity, **compact(result)})
                        save()
                        assert result["query_plan"]["origin"] == "deepseek", result["query_plan"]
                        print(f"DeepSeek plan {identity}: verified", flush=True)
                    for identity in ("summary-g1", "exercise-solve"):
                        result = cached[identity]
                        report["paid_api_attempts"] += 1
                        answer = rag.generate(result["question"], result["sources"], query_plan=result["query_plan"])
                        report["answers"].append({"id": identity, **answer})
                        save()
                        assert answer["status"] == "answered" and answer["citations"]
                        if identity == "exercise-solve":
                            assert "22" in answer["answer"] and "24" in answer["answer"], answer["answer"]
                        print(f"DeepSeek answer {identity}: verified", flush=True)
            assert digest(manifest(conn)) == before
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        after = digest(manifest(conn))
    assert before == after
    report.update(corpus_before_sha256=before, corpus_after_sha256=after, corpus_unchanged=True,
                  database_read_only=True, rules_passed=len(report["rules_cases"]),
                  llm_plans_verified=len(report["llm_cases"]), answers_verified=len(report["answers"]))
    save()
    print(json.dumps({key: report[key] for key in ("rules_passed", "llm_plans_verified", "answers_verified", "paid_api_attempts", "corpus_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
