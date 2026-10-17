"""Explicit live integration checks. --with-llm makes three billed API requests."""

import argparse
from datetime import datetime, timezone
import json

from fastapi.testclient import TestClient

from edu_rag.config import PROJECT_ROOT
from edu_rag.db import database_status, get_engine
from edu_rag.evaluate import main as evaluate_retrieval
from edu_rag.ingest import embed_pending, import_dataset, load_dataset
from edu_rag.main import app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--check-idempotence", action="store_true")
    args = parser.parse_args()
    engine = get_engine()
    before = database_status(engine)
    if args.check_idempotence:
        import_dataset(engine, load_dataset())
        indexed = embed_pending(engine)
        after = database_status(engine)
        assert before == after, "Repeated import changed record counts"
        assert indexed["embedded_this_run"] == 0, "Unchanged content was re-embedded"
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "database": before,
              "idempotence_verified": args.check_idempotence, "answers": []}
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health/db").status_code == 200
        response = client.get("/api/v1/courses")
        assert response.status_code == 200
        course = next(c for c in response.json() if c["code"] == "DEMO-CN-G3-MATH")
        topics = client.get("/api/v1/knowledge-points", params={"course_id": course["id"]}).json()
        assert len(topics) >= 16
        area = next(t for t in topics if t["code"] == "rectangle-area")
        detail = client.get("/api/v1/knowledge-points/" + area["id"]).json()
        assert len(detail["cards"]) >= 1 and len(detail["exercises"]) >= 3
        filtered = client.post("/api/v1/search", json={"question": "解释面积", "course_id": course["id"],
                                                     "knowledge_point_id": area["id"], "top_k": 3})
        assert filtered.status_code == 200
        assert filtered.json()["sources"]
        assert all(s["knowledge_point_id"] == area["id"] for s in filtered.json()["sources"])
        empty = client.post("/api/v1/ask", json={"question": "解释面积", "course_id": "ffffffff-ffff-4fff-8fff-ffffffffffff"})
        assert empty.status_code == 200 and empty.json()["model"] is None
        if args.with_llm:
            cases = [
                ("长方形长8厘米、宽3厘米，周长和面积分别是多少？", "answered", ["22", "24"]),
                ("53个人乘船，每条船最多坐8人，至少需要几条船？", "answered", ["7"]),
                ("化学中的氧化还原反应应该怎样配平？", "insufficient_evidence", []),
            ]
            for question, expected_status, numbers in cases:
                result = client.post("/api/v1/ask", json={"question": question, "course_id": course["id"]})
                assert result.status_code == 200, result.text
                answer = result.json()
                assert answer["status"] == expected_status, answer["answer"]
                assert all(number in answer["answer"] for number in numbers), answer["answer"]
                if expected_status == "answered":
                    assert answer["citations"]
                report["answers"].append({"question": question, "answer": answer["answer"],
                                         "status": answer["status"], "usage": answer["usage"],
                                         "citations": [{"citation": c["citation"], "source_title": c["source_title"]}
                                                       for c in answer["citations"]]})
    evaluate_retrieval()
    (PROJECT_ROOT / "data/v1/integration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "live_api_cases": len(report["answers"]),
                      "idempotence_verified": args.check_idempotence}, ensure_ascii=False))


if __name__ == "__main__":
    main()
