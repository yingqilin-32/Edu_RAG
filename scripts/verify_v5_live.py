"""Live HTTP checks for V5 answers, clarification, refusal and cost-budget stops."""

from datetime import datetime, timezone
import json
from time import perf_counter

import httpx

from edu_rag.config import PROJECT_ROOT
from evaluate_v4 import write_json


def main():
    cases = [
        {"id": "simple", "question": "三年级数学：长方形长9厘米宽4厘米，周长和面积分别是多少？", "statuses": ["answered"], "facts": ["26", "36"]},
        {"id": "complex", "question": "三年级数学：长方形周长34厘米，宽7厘米，先求长，再求面积。", "statuses": ["answered"], "facts": ["10", "70"]},
        {"id": "missing_width", "question": "长方形的长是9厘米，求面积。", "statuses": ["needs_clarification"]},
        {"id": "unknown_provenance", "question": "正方形周长公式最早由谁提出，文献年份是什么？", "statuses": ["refused", "needs_clarification"]},
        {"id": "scope", "question": "请解释初三一元二次方程配方法。", "statuses": ["refused", "needs_clarification"]},
        {"id": "call_cap", "question": "三年级长方形面积怎么算？", "max_model_calls": 1, "statuses": ["budget_exceeded"]},
        {"id": "token_cap", "question": "三年级长方形面积怎么算？", "max_total_tokens": 1000, "statuses": ["budget_exceeded"]},
        {"id": "preview", "question": "给我2道三年级有余数除法计算题。", "endpoint": "search", "statuses": ["exercises_found"]},
    ]
    rows = []
    with httpx.Client(timeout=190) as client:
        health = client.get("http://127.0.0.1:8000/health")
        health.raise_for_status()
        assert health.json()["stage"] == "V5"
        for case in cases:
            body = {"question": case["question"], "query_mode": "deepseek", "max_retries": 2,
                    "max_model_calls": case.get("max_model_calls", 10), "max_total_tokens": case.get("max_total_tokens", 30000)}
            start = perf_counter()
            response = client.post("http://127.0.0.1:8000/api/v5/"+case.get("endpoint", "ask"), json=body)
            response.raise_for_status()
            result = response.json()
            budget = result["workflow"]["budget"]
            passed = (result["status"] in case["statuses"] and budget["model_calls"] <= body["max_model_calls"] and
                      all(f in result["answer"] for f in case.get("facts", [])))
            if case["id"] == "preview":
                passed = passed and all(c["stage"] not in ("generate", "verify") for c in result["workflow"]["calls"])
            rows.append({"id": case["id"], "request": body, "passed": passed,
                         "http_elapsed_ms": round((perf_counter()-start)*1000, 3), "response": result})
            print(json.dumps({"id": case["id"], "status": result["status"], "passed": passed,
                              "calls": budget["model_calls"], "tokens": budget["actual_total_tokens"]}), flush=True)
    report = {"verified_at": datetime.now(timezone.utc).isoformat(), "health": health.json(),
              "all_passed": all(r["passed"] for r in rows), "results": rows,
              "note": "Small live HTTP integration regression, not an independent performance or accuracy sample."}
    write_json(PROJECT_ROOT/"data/v5/live_integration.json", report)
    assert report["all_passed"]


if __name__ == "__main__": main()
