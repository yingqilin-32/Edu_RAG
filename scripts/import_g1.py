"""Import reviewed, authored grade-one supplements and verify real retrieval."""

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json

from fastapi.testclient import TestClient
from sqlalchemy import text

from edu_rag.config import PROJECT_ROOT
from edu_rag.db import database_status, get_engine
from edu_rag.ingest import embed_pending, import_dataset, load_dataset, prepare_dataset
from edu_rag import knowledge, rag
from edu_rag.main import app
from edu_rag.scope import CurriculumScope


def grade_three_snapshot(engine):
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT ch.id,ch.content,ch.embedding::text,ch.embedding_model FROM public.edu_chunks ch "
            "JOIN public.edu_courses c ON c.id=ch.course_id WHERE c.code='DEMO-CN-G3-MATH' ORDER BY ch.id"
        )).all()
    return sha256(json.dumps([list(map(str, row)) for row in rows]).encode()).hexdigest()


def import_all(engine, datasets, prepared):
    # Validate both files before this transaction, then commit both courses together.
    with engine.begin() as connection:
        class AtomicImport:
            @contextmanager
            def begin(self):
                yield connection
        return [import_dataset(AtomicImport(), dataset, prepared=chunks)
                for dataset, chunks in zip(datasets, prepared, strict=True)]


def main():
    datasets = [load_dataset(PROJECT_ROOT / f"data/g1/math_{semester}.json") for semester in ("first", "second")]
    prepared = [prepare_dataset(dataset) for dataset in datasets]
    engine = get_engine()
    original = grade_three_snapshot(engine)
    imported = import_all(engine, datasets, prepared)
    indexed = embed_pending(engine)
    import_all(engine, datasets, prepared)
    assert embed_pending(engine)["embedded_this_run"] == 0
    assert grade_three_snapshot(engine) == original, "Original grade-three text or vectors changed"

    rows = knowledge.courses(engine, CurriculumScope(grade=1))
    courses = [next(row for row in rows if row["code"] == dataset.course_code) for dataset in datasets]
    for dataset, course in zip(datasets, courses, strict=True):
        assert course["knowledge_points"] == len(dataset.topics)
        assert course["exercises"] == sum(len(topic.exercises) for topic in dataset.topics)
        assert course["chunks"] == course["embedded_chunks"]
        topics = knowledge.knowledge_points(engine, scope=CurriculumScope(course_id=course["id"]))
        detail = knowledge.knowledge_detail(engine, topics[0]["id"])
        assert detail["cards"][0]["metadata"]["reference_sources"][0]["url"] == str(dataset.reference_sources[0].url)

    cases = [
        ("9+6可以怎样凑十计算？", "first", "addition-subtraction-within-20"),
        ("4元3角是多少角？", "second", "money-units"),
        ("73是几个十和几个一组成的？", "second", "numbers-within-100"),
    ]
    retrieval = []
    edition = datasets[0].course.textbook_edition
    for question, semester, topic in cases:
        result = rag.search(rag.SearchRequest(question=question, grade=1, semester=semester,
                                            textbook_edition=edition, top_k=5))
        assert result["sources"]
        assert all(source["grade"] == 1 and source["semester"] == semester
                   and source["textbook_edition"] == edition for source in result["sources"])
        rank = next((i for i, source in enumerate(result["sources"], 1) if source["knowledge_point_code"] == topic), None)
        assert rank is not None, question
        assert all(source["metadata"]["content_origin"] == "authored" and source["metadata"]["reference_sources"]
                   for source in result["sources"])
        retrieval.append({"question": question, "semester": semester, "expected_topic_rank": rank,
                          "sources_within_scope": True})

    with TestClient(app) as client:
        result = client.get("/api/v2/courses", params={"grade": 1})
        assert result.status_code == 200
        assert {row["code"] for row in result.json()} >= {dataset.course_code for dataset in datasets}
        for body in [{"grade": 1, "semester": "full_year"}, {"grade": 3, "course_id": str(courses[0]["id"])}]:
            response = client.post("/api/v2/ask", json={"question": "解释数的组成", **body})
            assert response.status_code == 200, response.text
            assert response.json()["model"] is None and response.json()["sources"] == []

    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "database": database_status(engine),
              "courses": [{key: str(course[key]) if key == "id" else course[key]
                           for key in ("id", "code", "title", "grade", "semester", "textbook_edition",
                                       "knowledge_points", "exercises", "chunks", "embedded_chunks")}
                          for course in courses],
              "import": imported, "embedded_this_run": indexed["embedded_this_run"],
              "idempotence_verified": True, "grade_three_vectors_unchanged": True,
              "reference_metadata_preserved": True, "empty_or_conflicting_scope_skips_model": True,
              "retrieval": retrieval, "paid_model_calls": 0}
    (PROJECT_ROOT / "data/g1/import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry_path = PROJECT_ROOT / "data/sources/sohu_431916810_99911694.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry.update({"review_status": "topic_reference_processed", "ingested_into_database": True,
                     "original_article_body_ingested": False, "ingestion_mode": "topic_reference_with_authored_lessons",
                     "ingested_at_utc": report["verified_at_utc"],
                     "derived_datasets": [{"dataset_id": dataset.dataset_id, "course_code": dataset.course_code,
                                           "source_uri": dataset.source_uri} for dataset in datasets],
                     "review_resolution": "参考主题目录后重新编写讲解与练习；原网页的歧义算式、未核验表述和图片依赖题未直接导入。"})
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
