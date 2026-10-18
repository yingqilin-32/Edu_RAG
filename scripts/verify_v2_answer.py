"""One explicitly enabled, billed end-to-end check against the running backend."""

import argparse
from datetime import datetime, timezone
import json

import httpx

from edu_rag.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="明确启用一次付费模型请求")
    args = parser.parse_args()
    if not args.with_llm:
        parser.error("需要 --with-llm 才会调用真实模型。")
    scope = {"grade": 3, "textbook_edition": "自编演示资料", "semester": "full_year", "subject": "math"}
    question = "长方形长8厘米、宽3厘米，周长和面积分别是多少？"
    response = httpx.post("http://127.0.0.1:8000/api/v2/ask", json={"question": question, **scope}, timeout=180)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "answered" and result["citations"], result["answer"]
    assert "22" in result["answer"] and "24" in result["answer"], result["answer"]
    assert all(all(source[key] == value for key, value in scope.items()) for source in result["sources"])
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "scope": scope,
              "question": question, "status": result["status"], "model": result["model"],
              "answer": result["answer"], "usage": result["usage"], "all_sources_match_scope": True,
              "citations": [{key: source[key] for key in ("citation", "source_title", "grade", "textbook_edition", "semester")}
                            for source in result["citations"]]}
    (PROJECT_ROOT / "data/v2/answer_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
