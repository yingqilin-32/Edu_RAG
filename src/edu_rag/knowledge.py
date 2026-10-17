"""Read the curriculum catalog independently from top-k semantic retrieval."""

from sqlalchemy import text
from sqlalchemy.engine import Engine


def courses(engine: Engine) -> list[dict]:
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        return [dict(row) for row in conn.execute(text(
            "SELECT c.*, (SELECT count(*) FROM public.edu_knowledge_points k WHERE k.course_id=c.id) AS knowledge_points, "
            "(SELECT count(*) FROM public.edu_exercises e WHERE e.course_id=c.id) AS exercises, "
            "(SELECT count(*) FROM public.edu_chunks ch WHERE ch.course_id=c.id) AS chunks, "
            "(SELECT count(*) FROM public.edu_chunks ch WHERE ch.course_id=c.id AND ch.embedding IS NOT NULL) AS embedded_chunks "
            "FROM public.edu_courses c ORDER BY c.code"
        )).mappings()]


def knowledge_points(engine: Engine, course_id=None) -> list[dict]:
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        return [dict(row) for row in conn.execute(text(
            "SELECT k.*, (SELECT count(*) FROM public.edu_exercise_knowledge_points ek "
            "WHERE ek.knowledge_point_id=k.id) AS exercises "
            "FROM public.edu_knowledge_points k "
            "WHERE (CAST(:course AS uuid) IS NULL OR k.course_id=CAST(:course AS uuid)) "
            "ORDER BY k.sort_order,k.code"
        ), {"course": str(course_id) if course_id else None}).mappings()]


def knowledge_detail(engine: Engine, knowledge_point_id) -> dict | None:
    params = {"id": str(knowledge_point_id)}
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        topic = conn.execute(text(
            "SELECT * FROM public.edu_knowledge_points WHERE id=CAST(:id AS uuid)"
        ), params).mappings().first()
        if topic is None:
            return None
        cards = [dict(row) for row in conn.execute(text(
            "SELECT c.id,c.content,c.chunk_index,d.title,d.source_uri,d.license_note, "
            "(c.embedding IS NOT NULL) AS indexed "
            "FROM public.edu_chunks c JOIN public.edu_documents d ON d.id=c.document_id "
            "WHERE c.knowledge_point_id=CAST(:id AS uuid) ORDER BY d.source_key,c.chunk_index"
        ), params).mappings()]
        exercises = [dict(row) for row in conn.execute(text(
            "SELECT e.id,e.code,e.stem,e.question_type,e.difficulty,e.options,e.answer,e.explanation "
            "FROM public.edu_exercises e JOIN public.edu_exercise_knowledge_points ek ON ek.exercise_id=e.id "
            "WHERE ek.knowledge_point_id=CAST(:id AS uuid) ORDER BY e.code"
        ), params).mappings()]
        return {**dict(topic), "cards": cards, "exercises": exercises}
