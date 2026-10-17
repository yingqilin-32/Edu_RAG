"""Small development-set retrieval baseline; no LLM calls or answer grading."""

import json
from time import perf_counter

from edu_rag.config import PROJECT_ROOT
from edu_rag.rag import SearchRequest, search


def main():
    cases = json.loads((PROJECT_ROOT / "data/v1/eval.json").read_text(encoding="utf-8"))
    cases = [case for case in cases if case["topic"] is not None]
    results = []
    for case in cases:
        start = perf_counter()
        sources = search(SearchRequest(question=case["question"]))["sources"]
        rank = next((i for i, source in enumerate(sources, 1) if source["knowledge_point_code"] == case["topic"]), None)
        results.append({**case, "rank": rank, "seconds": round(perf_counter() - start, 3),
                        "retrieved_topics": [s["knowledge_point_code"] for s in sources]})
    report = {
        "set": "original-g3-development-v1", "queries": len(results),
        "knowledge_point_hit_at_5": sum(r["rank"] is not None for r in results) / len(results),
        "knowledge_point_mrr_at_5": sum(1 / r["rank"] if r["rank"] else 0 for r in results) / len(results),
        "note": "Small in-domain development set, not a held-out benchmark. Negative cases require separate answer checks.",
        "results": results,
    }
    output = PROJECT_ROOT / "data/v1/retrieval_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
