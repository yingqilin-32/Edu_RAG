"""Live V2 verification; isolated scope fixtures are rolled back, no paid LLM calls."""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import text

from edu_rag.config import ConfigurationError, PROJECT_ROOT
from edu_rag.db import database_status, get_engine
from edu_rag.embedding import count_tokens, get_encoder
from edu_rag.evaluate import main as evaluate
from edu_rag.ingest import embed_pending, import_dataset, load_dataset, prepare_dataset
from edu_rag.main import app
from edu_rag import knowledge, rag
from edu_rag.scope import CurriculumScope


class TransactionEngine:
    """Keep production SQL in one rollback-only fixture transaction.

    SET TRANSACTION READ ONLY is skipped only here: the surrounding transaction
    must also write fixtures. Normal application connections remain read-only.
    """
    def __init__(self, connection):
        self.connection = connection

    @contextmanager
    def begin(self):
        yield self.connection

    @contextmanager
    def connect(self):
        connection = self.connection
        class Reader:
            def execute(self, statement, *args, **kwargs):
                if str(statement) == "SET TRANSACTION READ ONLY":
                    return None
                return connection.execute(statement, *args, **kwargs)
        yield Reader()


def vector_snapshot(engine):
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id,content,embedding::text,embedding_model FROM public.edu_chunks ORDER BY id")).all()
        return sha256(json.dumps([list(map(str, row)) for row in rows]).encode()).hexdigest()


def main():
    engine = get_engine()
    before = database_status(engine)
    original_vectors = vector_snapshot(engine)
    dataset = load_dataset()
    imported = import_dataset(engine, dataset)
    indexed = embed_pending(engine)
    import_dataset(engine, dataset)
    repeated = embed_pending(engine)
    assert repeated["embedded_this_run"] == 0
    assert database_status(engine) == before
    assert vector_snapshot(engine) == original_vectors, "Metadata-only upgrade changed vectors or indexed text"

    example = load_dataset(PROJECT_ROOT / "data/v2/cleaning_example.json")
    prepared = prepare_dataset(example, max_tokens=120, overlap_tokens=24)
    cards = prepared[0][0]
    assert len(cards) > 3
    assert all(count_tokens(chunk.content) <= 120 for chunk in cards)
    combined = "\n".join(chunk.content for chunk in cards)
    assert "Edu RAG 原创测试页眉" not in combined and "第1页" not in combined
    assert "$$\nP = (a + b) × 2\nS = a × b\n$$" in combined
    assert "| 长方形 | 8厘米 | 3厘米 | 22厘米 | 24平方厘米 |" in combined
    encoder = get_encoder()
    scope_results = []
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            isolated = TransactionEngine(conn)
            fixtures = []
            for suffix, grade, edition, semester in [
                ("G3-A-FIRST", 3, "V2 原创测试版 A", "first"),
                ("G4-A-FIRST", 4, "V2 原创测试版 A", "first"),
                ("G3-B-FIRST", 3, "V2 原创测试版 B", "first"),
                ("G3-A-SECOND", 3, "V2 原创测试版 A", "second"),
            ]:
                fixture = example.model_copy(deep=True)
                fixture.course_code = "TEST-V2-" + suffix
                fixture.course.grade, fixture.course.textbook_edition, fixture.course.semester = grade, edition, semester
                import_dataset(isolated, fixture, prepared=prepared)
                course_id = conn.scalar(text("SELECT id FROM public.edu_courses WHERE code=:code"), {"code": fixture.course_code})
                fixtures.append((fixture, course_id))
            fixture_index = embed_pending(isolated, encoder)
            for fixture, course_id in fixtures:
                scope = CurriculumScope(grade=fixture.course.grade, textbook_edition=fixture.course.textbook_edition,
                                        semester=fixture.course.semester, curriculum=fixture.course.curriculum)
                catalog = knowledge.courses(isolated, scope)
                assert [row["id"] for row in catalog] == [course_id]
                result = rag.search(rag.SearchRequest(question="长方形长8厘米、宽3厘米，周长和面积分别是多少？",
                                    top_k=1, **scope.model_dump()), engine=isolated, encoder=encoder)
                assert result["sources"] and all(source["course_id"] == course_id for source in result["sources"])
                topics = knowledge.knowledge_points(isolated, scope=scope)
                assert topics and all(topic["course_id"] == course_id for topic in topics)
                scope_results.append({**scope.model_dump(mode="json"), "returned_sources": len(result["sources"]), "isolated": True})

            with patch("edu_rag.main.get_engine", return_value=isolated), patch("edu_rag.rag.get_engine", return_value=isolated), \
                    patch("edu_rag.rag.generate", side_effect=AssertionError("Empty scope invoked paid model")):
                with TestClient(app) as client:
                    for body in [
                        {"grade": 12, "textbook_edition": "未收录测试版本"},
                        {"course_id": str(fixtures[1][1]), "grade": 3},
                        {"textbook_edition": "' OR 1=1 --"},
                    ]:
                        response = client.post("/api/v2/ask", json={"question": "解释长方形面积", **body})
                        assert response.status_code == 200, response.text
                        assert response.json()["model"] is None and response.json()["sources"] == []
                    topic_id = conn.scalar(text("SELECT id FROM public.edu_knowledge_points WHERE course_id=:id"), {"id": fixtures[1][1]})
                    assert client.get(f"/api/v2/knowledge-points/{topic_id}", params={"grade": 3}).status_code == 404
                    assert client.get("/api/v2/courses", params={"textbook_edition": "未收录测试版本"}).json() == []

            # Mislabeled import is refused without changing the existing course.
            mismatch = fixtures[0][0].model_copy(deep=True)
            mismatch.course.grade = 4
            try:
                import_dataset(isolated, mismatch, prepared=prepared)
                raise AssertionError("Mismatched course metadata was accepted")
            except ConfigurationError:
                pass

            # Shrinking a document removes its old derived chunks, retaining
            # stable IDs and the unchanged exercise vector.
            fixture, course_id = fixtures[0]
            document_id = conn.scalar(text("SELECT id FROM public.edu_documents WHERE course_id=:id"), {"id": course_id})
            old_chunk_id = conn.scalar(text("SELECT id FROM public.edu_chunks WHERE document_id=:id AND chunk_index=0"), {"id": document_id})
            short = fixture.model_copy(deep=True)
            short.topics[0].content = "长方形面积=长×宽。"
            import_dataset(isolated, short, prepared=prepare_dataset(short, max_tokens=120))
            new_chunks = conn.execute(text("SELECT id,embedding FROM public.edu_chunks WHERE document_id=:id"), {"id": document_id}).all()
            assert len(new_chunks) == 1 and new_chunks[0].id == old_chunk_id and new_chunks[0].embedding is None
            assert conn.scalar(text("SELECT status FROM public.edu_documents WHERE id=:id"), {"id": document_id}) == "parsed"
            assert conn.scalar(text("SELECT count(*) FROM public.edu_chunks WHERE course_id=:id AND exercise_id IS NOT NULL AND embedding IS NOT NULL"), {"id": course_id}) == 1
        finally:
            transaction.rollback()
    assert database_status(engine) == before, "Fixture records escaped rollback"
    assert vector_snapshot(engine) == original_vectors
    output = PROJECT_ROOT / "data/v2"
    evaluate(output=output / "retrieval_report.json")
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "database": database_status(engine),
        "import": imported, "upgrade_reembedded_chunks": indexed["embedded_this_run"],
        "idempotence_verified": True, "original_vectors_unchanged": True,
        "example": {"card_chunks": len(cards), "pages": sorted({c.source_page for c in cards}),
                    "max_tokens": max(count_tokens(c.content) for c in cards),
                    "formula_and_table_preserved": True, "duplicate_blocks_removed": cards[0].metadata["duplicate_blocks"]},
        "scope_cases": scope_results, "fixture_embedded_chunks": fixture_index["embedded_this_run"],
        "empty_scope_skips_model": True, "conflicting_scope_returns_empty": True,
        "metadata_mismatch_refused": True, "stale_chunks_removed": True,
        "fixture_transaction_rolled_back": True, "paid_model_calls": 0,
    }
    (output / "integration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
