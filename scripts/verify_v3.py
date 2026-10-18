"""Read-only, fixed-snapshot Dense/BM25/RRF comparison; never calls an LLM."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from statistics import median
from time import perf_counter
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import text

from edu_rag import __version__, rag
from edu_rag.config import PROJECT_ROOT, get_settings
from edu_rag.db import get_engine
from edu_rag.embedding import get_encoder
from edu_rag.main import app
from edu_rag.retrieval import BM25_B, BM25_K1, TOKENIZER_VERSION
from edu_rag.scope import SCOPE_SQL

MODES = ("dense", "bm25", "hybrid")


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def manifest(conn):
    rows = conn.execute(text(
        "SELECT m.id,c.id AS course_id,c.code AS course_code,c.grade,c.semester,c.textbook_edition,c.subject,c.curriculum,"
        "k.code AS topic_code,e.code AS exercise_code,coalesce(d.title,e.code) AS title,m.content,m.metadata,"
        "m.embedding::text AS embedding,m.embedding_model,e.answer,e.explanation "
        "FROM public.edu_chunks m JOIN public.edu_courses c ON c.id=m.course_id "
        "LEFT JOIN public.edu_knowledge_points k ON k.id=m.knowledge_point_id "
        "LEFT JOIN public.edu_exercises e ON e.id=m.exercise_id "
        "LEFT JOIN public.edu_documents d ON d.id=m.document_id ORDER BY m.id"
    )).mappings().all()
    items = []
    for row in rows:
        item = dict(row)
        for key in ("id", "course_id"):
            item[key] = str(item[key])
        for key in ("content", "metadata", "embedding", "answer", "explanation"):
            item[key + "_sha256"] = digest(item.pop(key))
        items.append(item)
    return items


class SnapshotEngine:
    """Production search SQL on an already read-only repeatable-read transaction."""
    def __init__(self, conn):
        self.conn = conn

    @contextmanager
    def connect(self):
        conn = self.conn
        class Reader:
            def execute(self, statement, *args, **kwargs):
                if str(statement) == "SET TRANSACTION READ ONLY":
                    return None
                return conn.execute(statement, *args, **kwargs)
        yield Reader()


class RecordingEncoder:
    def __init__(self, encoder):
        self.encoder, self.model_id, self.last_vector = encoder, encoder.model_id, None

    def encode(self, texts):
        values = self.encoder.encode(texts)
        self.last_vector = values[0]
        return values


def relevant(case, source):
    key = "exercise_code" if "exercise_code" in case else "knowledge_point_code"
    return source[key] == case.get("exercise_code", case.get("topic"))


def metrics(rows):
    count = len(rows)
    ranks = [row["rank"] for row in rows]
    return {"queries": count, "hit_at_1": sum(rank == 1 for rank in ranks) / count,
            "hit_at_5": sum(rank is not None for rank in ranks) / count,
            "mrr_at_5": sum(1 / rank if rank else 0 for rank in ranks) / count,
            "median_total_ms": median(row["retrieval"]["timing_ms"]["total"] for row in rows)}


def validate_result(result, scope):
    info, sources = result["retrieval"], result["sources"]
    assert len({str(source["id"]) for source in sources}) == len(sources)
    ranks = {branch: {row["id"]: row["rank"] for row in info["candidate_lists"][branch]} for branch in ("dense", "bm25")}
    for i, source in enumerate(sources, 1):
        assert source["citation"] == f"C{i}"
        assert all(str(source[key]) == str(scope[key]) for key in scope)
        if info["mode"] == "hybrid":
            expected = sum(1 / (info["rrf_k"] + ranks[branch][str(source["id"])])
                           for branch in ranks if str(source["id"]) in ranks[branch])
            assert abs(expected - source["rrf_score"]) < 1e-12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=str, default="data/v3/eval.json")
    parser.add_argument("--output", type=str, default="data/v3/retrieval_report.json")
    args = parser.parse_args()
    eval_path = PROJECT_ROOT / args.eval
    suite = json.loads(eval_path.read_text(encoding="utf-8"))
    print(f"Loading local encoder; {len(suite['cases'])} frozen cases × 3 modes", flush=True)
    warmup_start = perf_counter()
    encoder = RecordingEncoder(get_encoder())
    encoder.encode(["检索模型预热"])
    warmup_ms = (perf_counter() - warmup_start) * 1000
    engine = get_engine()
    results = []
    checks = []
    with engine.connect().execution_options(isolation_level="REPEATABLE READ") as conn:
        with conn.begin():
            conn.execute(text("SET TRANSACTION READ ONLY"))
            assert conn.scalar(text("SHOW transaction_read_only")) == "on"
            corpus = manifest(conn)
            before = digest(corpus)
            snapshot_engine = SnapshotEngine(conn)
            scopes = {item["course_code"]: {key: item[key] for key in
                      ("course_id", "grade", "semester", "textbook_edition", "subject", "curriculum")} for item in corpus}
            for i, case in enumerate(suite["cases"]):
                scope = scopes[case["course_code"]]
                # Check gold labels before scoring; a typo must fail rather than depress metrics.
                assert any(item["course_code"] == case["course_code"] and
                           (item["exercise_code"] == case["exercise_code"] if "exercise_code" in case else item["topic_code"] == case["topic"])
                           for item in corpus), f"Missing gold label: {case['id']}"
                # Rotate modes to reduce systematic first-query timing bias; no query vector cache.
                modes = MODES[i % 3:] + MODES[:i % 3]
                modes_results = {}
                for mode in modes:
                    request = rag.SearchRequest(question=case["question"], **scope, retrieval_mode=mode, top_k=5, candidate_k=20, rrf_k=60)
                    result = rag.search(request, engine=snapshot_engine, encoder=encoder)
                    validate_result(result, scope)
                    if mode == "dense":
                        # Independent old V2 SQL top-5 check, same vector and scope.
                        ids = conn.scalars(text(
                            "SELECT m.id FROM public.edu_chunks m JOIN public.edu_courses c ON c.id=m.course_id "
                            "WHERE m.embedding IS NOT NULL AND m.embedding_model=:model AND " + SCOPE_SQL +
                            " ORDER BY m.embedding OPERATOR(extensions.<=>) CAST(:vector AS extensions.vector),m.id LIMIT 5"
                        ), {**request.sql_params(), "model": encoder.model_id, "vector": json.dumps(encoder.last_vector)}).all()
                        assert [s["id"] for s in result["sources"]] == ids
                    rank = next((i for i, source in enumerate(result["sources"], 1) if relevant(case, source)), None)
                    selected_fields = ("id", "course_id", "knowledge_point_code", "exercise_code", "source_title", "citation",
                                       "similarity", "dense_rank", "bm25_rank", "bm25_score", "rrf_score")
                    modes_results[mode] = {"rank": rank, "retrieval": result["retrieval"],
                                          "top_5": [{key: source[key] for key in selected_fields} for source in result["sources"]]}
                results.append({**case, "scope": scope, "modes": modes_results})
                if (i+1) % 5 == 0:
                    print(f"Compared {i+1}/{len(suite['cases'])} cases", flush=True)

            # Real API boundary checks; patch only the connection source, never ranking.
            g1 = scopes["DEMO-CN-G1-MATH-FIRST"]
            with patch("edu_rag.rag.get_engine", return_value=snapshot_engine), \
                 patch("edu_rag.rag.get_encoder", return_value=encoder), \
                 patch("edu_rag.rag.generate", side_effect=AssertionError("Verification must not call a paid model")):
                with TestClient(app) as client:
                    for scope in [dict(g1, grade=3), dict(g1, semester="second"),
                                  dict(g1, textbook_edition="未收录版本"), {"grade": 12},
                                  {"course_id": "00000000-0000-4000-8000-000000000000"}]:
                        for mode in MODES:
                            response = client.post("/api/v3/ask", json={"question": "解释数的组成", **scope, "retrieval_mode": mode})
                            assert response.status_code == 200, response.text
                            assert response.json()["sources"] == [] and response.json()["model"] is None
                        checks.append({"scope": scope, "modes": list(MODES), "empty_and_skips_model": True})
                    response = client.post("/api/v3/search", json={"question": "zzznomatchingtermv3", **g1, "retrieval_mode": "bm25"})
                    assert response.status_code == 200 and response.json()["sources"] == []
                    positive = client.post("/api/v3/search", json={"question": "9+6怎样凑十？", **g1})
                    assert positive.status_code == 200
                    validate_result(positive.json(), g1)
                    assert positive.json()["retrieval"]["mode"] == "hybrid"
            assert digest(manifest(conn)) == before

    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        after = digest(manifest(conn))
    assert after == before, "Corpus changed during the comparison; discard this run"
    summary = {}
    for group in ("all", "natural-topic", "exact-exercise-id"):
        summary[group] = {mode: metrics([row["modes"][mode] for row in results if group == "all" or row["group"] == group])
                          for mode in MODES}
    changes = {kind: [row["id"] for row in results if predicate(row["modes"]["dense"]["rank"] or 6,
                                                               row["modes"]["hybrid"]["rank"] or 6)]
               for kind, predicate in (("improved", lambda d, h: h < d), ("regressed", lambda d, h: h > d),
                                       ("unchanged", lambda d, h: h == d))}
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "version": __version__, "set": suite["set"],
              "evaluation_sha256": sha256(eval_path.read_bytes()).hexdigest(), "note": suite["note"],
              "configuration": {"embedding_model": encoder.model_id, "top_k": 5, "candidate_k": 20, "rrf_k": 60,
                                "rrf_weights": {"dense": 1, "bm25": 1}, "bm25_k1": BM25_K1, "bm25_b": BM25_B,
                                "tokenizer": TOKENIZER_VERSION, "bm25_max_documents": get_settings().bm25_max_documents},
              "corpus": {"chunks": len(corpus), "courses": len(scopes), "before_sha256": before, "after_sha256": after,
                         "unchanged": True, "manifest_file": "corpus_manifest.json"},
              "timing": {"warmup_ms_excluded": round(warmup_ms, 3), "mode_order_rotated": True, "query_vectors_cached": False,
                         "note": "Single warmed CPU execution per query/mode, includes network database time; not a load benchmark. BM25 skips neural inference."},
              "summary": summary, "hybrid_vs_dense": changes,
              "verification": {"legacy_dense_sql_top_5_matches_all_cases": True, "rrf_scores_recomputed": True,
                               "citations_assigned_after_fusion": True, "scope_and_no_model_checks": checks,
                               "bm25_zero_overlap_empty": True, "hybrid_default_api_verified": True,
                               "database_read_only": True, "paid_model_calls": 0}, "results": results}
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name("corpus_manifest.json").write_text(json.dumps(corpus, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "hybrid_vs_dense": changes, "corpus_unchanged": True, "paid_model_calls": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
