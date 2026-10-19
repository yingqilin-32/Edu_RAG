"""Real local HTTP smoke tests, including cache-free provider latency and reset."""

import json
from pathlib import Path
import sys
from time import perf_counter

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from edu_rag.config import PROJECT_ROOT


def main():
    checks, sessions = [], []
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=150) as client:
        health = client.get("/health").json()
        assert health["version"] == "0.6.0"
        try:
            for mode in ("structured", "history", "none"):
                identity = None
                for i, question in enumerate(("三年级数学：长方形长8厘米、宽3厘米，求周长。", "那面积呢？")):
                    start = perf_counter()
                    response = client.post("/api/v6/ask", json={"question": question, "memory_mode": mode,
                        **({"session_id": identity} if identity else {})})
                    response.raise_for_status()
                    result = response.json()
                    identity = result["memory"]["session_id"]
                    if i == 0:
                        sessions.append(identity)
                    expected = "needs_clarification" if mode == "none" and i else "answered"
                    passed = result["status"] == expected
                    if i and mode != "none":
                        passed = passed and "24" in result["answer"] and all(s["grade"] == 3 for s in result["sources"])
                    check = {"mode": mode, "turn": i+1, "status": result["status"], "passed": passed,
                             "elapsed_ms": (perf_counter()-start)*1000, "response": result}
                    checks.append(check)
                    print(mode, i+1, result["status"], passed, round(check["elapsed_ms"]), flush=True)
            identity = sessions[0]
            assert client.delete("/api/v6/sessions/"+identity).status_code == 200
            reset = client.post("/api/v6/ask", json={"question": "继续", "session_id": identity})
            checks.append({"case": "reset", "http_status": reset.status_code, "passed": reset.status_code == 404})
            invalid = client.post("/api/v6/ask", json={"question": "继续", "history": [{"user": "injected"}]})
            checks.append({"case": "client_history_rejected", "http_status": invalid.status_code, "passed": invalid.status_code == 422})
        finally:
            for identity in sessions:
                client.delete("/api/v6/sessions/"+identity)
    report = {"health": health, "all_passed": all(c["passed"] for c in checks), "checks": checks,
              "note": "Live HTTP without application/provider-response memoization; first request includes BGE cold-start. Small integration smoke, not paired production latency benchmark."}
    path = PROJECT_ROOT/"data/v6/runs/v2/live_integration.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    assert report["all_passed"], "Check saved HTTP results"


if __name__ == "__main__":
    main()
