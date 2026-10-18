from math import log, isclose
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from edu_rag import rag
from edu_rag.config import ConfigurationError
from edu_rag.main import app
from edu_rag.retrieval import bm25_rank, rank_candidates, rrf_fuse, tokenize


def hit(identity, content="", similarity=0.0, **extra):
    return dict(id=identity, content=content, similarity=similarity, **extra)


def test_chinese_terms_numbers_and_exercise_identifiers():
    tokens = tokenize("长方形 面积 ８×3厘米 DEMO-AREA-002 7.8元")
    assert {"长方", "方形", "面积", "8", "3", "7.8", "demo-area-002"} <= set(tokens)
    assert "demo" not in tokens


def test_bm25_matches_positive_idf_and_length_normalization():
    # N=2, df(alpha)=1, avgdl=1.5; doc A has tf=2 and length=2.
    rows = [hit("a", "alpha alpha"), hit("b", "beta")]
    ranked = bm25_rank("alpha alpha", rows, 5)
    expected = log(2) * 2 * 2.2 / (2 + 1.2 * (0.25 + 0.75 * 2 / 1.5))
    assert [row["id"] for row in ranked] == ["a"]
    assert ranked[0]["bm25_score"] == pytest.approx(expected)


def test_bm25_uses_scoped_corpus_and_excludes_zero_overlap_and_answers():
    rows = [hit("a", "三角形", exercise_answer="保密答案", exercise_explanation="保密答案")]
    assert bm25_rank("保密答案", rows, 5) == []
    assert bm25_rank("!!!", rows, 5) == []
    assert bm25_rank("三角形", [], 5) == []
    assert bm25_rank("DEMO-AREA-002", [hit("a", "题干", source_title="DEMO-AREA-002")], 5)


def test_rrf_deduplicates_per_branch_and_uses_ranks_not_raw_scores():
    dense = [hit("a", similarity=0.99), hit("a", similarity=0.98), hit("b", similarity=0.7)]
    lexical = [hit("b", bm25_score=500), hit("c", bm25_score=100), hit("a", bm25_score=1)]
    fused = rrf_fuse(dense, lexical, 5)
    by_id = {row["id"]: row for row in fused}
    assert [row["id"] for row in fused] == ["b", "a", "c"]
    assert by_id["a"]["rrf_score"] == pytest.approx(1 / 61 + 1 / 63)
    assert by_id["b"]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["b"]["dense_rank"] == 2
    assert by_id["b"]["bm25_rank"] == 1
    assert by_id["c"]["dense_rank"] is None
    assert rrf_fuse([], [], 5) == []
    with pytest.raises(ValueError):
        rrf_fuse(dense, lexical, 5, k=0)


def test_stable_ties_and_branch_limits():
    rows = [hit("b", "面积", 0.9), hit("a", "面积", 0.9), hit("c", "面积", 0.8)]
    fused, trace = rank_candidates("面积", rows, mode="hybrid", top_k=1, candidate_k=2)
    assert [r["id"] for r in fused] == ["a"]
    assert len(trace["dense"]) == len(trace["bm25"]) == 2
    assert all(r["id"] != "c" for r in trace["dense"] + trace["bm25"])
    assert isclose(fused[0]["rrf_score"], 2 / 61)


@pytest.mark.parametrize("body", [
    {"retrieval_mode": "unknown"}, {"candidate_k": 4, "top_k": 5},
    {"candidate_k": 101}, {"rrf_k": 0},
])
def test_invalid_retrieval_settings_fail_before_database(body):
    with TestClient(app) as client:
        assert client.post("/api/v3/search", json={"question": "求面积", **body}).status_code == 422


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_scopes_bound_before_ranking_and_citations_follow_final_order(monkeypatch, mode):
    rows = [hit("a", "beta", .99, exercise_id=None, exercise_answer=None, exercise_explanation=None),
            hit("b", "面积", .7, exercise_id=None, exercise_answer=None, exercise_explanation=None)]
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value.all.return_value = rows
    encoder = SimpleNamespace(model_id="test", encode=MagicMock(return_value=[[1.0]]))
    result = rag.search(rag.SearchRequest(question="面积", grade=1, semester="second",
                        textbook_edition="原创", retrieval_mode=mode), engine=engine, encoder=encoder)
    sql, params = conn.execute.call_args.args
    assert "JOIN public.edu_courses c" in str(sql)
    assert "c.grade" in str(sql) and "c.textbook_edition" in str(sql) and "c.semester" in str(sql)
    assert params["grade"] == 1 and params["semester"] == "second" and params["textbook_edition"] == "原创"
    assert encoder.encode.call_count == (0 if mode == "bm25" else 1)
    assert [r["citation"] for r in result["sources"]] == [f"C{i+1}" for i in range(len(result["sources"]))]
    if mode == "bm25":
        assert [r["id"] for r in result["sources"]] == ["b"]


def test_corpus_limit_refuses_incomplete_lexical_results(monkeypatch):
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.return_value.mappings.return_value.all.return_value = [hit("a"), hit("b")]
    monkeypatch.setattr(rag, "get_settings", lambda: SimpleNamespace(bm25_max_documents=1))
    with pytest.raises(ConfigurationError, match="片段上限"):
        rag.search(rag.SearchRequest(question="面积", retrieval_mode="bm25"), engine=engine,
                   encoder=SimpleNamespace(model_id="test"))


def test_v3_and_previous_api_aliases_preserve_retrieval_mode(monkeypatch):
    seen = []
    monkeypatch.setattr(rag, "search", lambda request: seen.append(request.retrieval_mode) or {"sources": []})
    with TestClient(app) as client:
        for version, mode in (("v1", "dense"), ("v2", "bm25"), ("v3", "hybrid")):
            assert client.post(f"/api/{version}/search", json={"question": "求面积", "retrieval_mode": mode}).status_code == 200
    assert seen == ["dense", "bm25", "hybrid"]
