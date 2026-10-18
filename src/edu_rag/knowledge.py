"""Read the curriculum catalog independently from top-k semantic retrieval."""

from sqlalchemy import text
from sqlalchemy.engine import Engine
from edu_rag.scope import CurriculumScope, SCOPE_SQL


def courses(engine: Engine, scope: CurriculumScope | None = None) -> list[dict]:
    scope = scope or CurriculumScope()
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        return [dict(row) for row in conn.execute(text(
            "SELECT c.*, (SELECT count(*) FROM public.edu_knowledge_points k WHERE k.course_id=c.id) AS knowledge_points, "
            "(SELECT count(*) FROM public.edu_exercises e WHERE e.course_id=c.id) AS exercises, "
            "(SELECT count(*) FROM public.edu_chunks ch WHERE ch.course_id=c.id) AS chunks, "
            "(SELECT count(*) FROM public.edu_chunks ch WHERE ch.course_id=c.id AND ch.embedding IS NOT NULL) AS embedded_chunks "
            "FROM public.edu_courses c WHERE " + SCOPE_SQL + " ORDER BY c.code"
        ), scope.sql_params()).mappings()]


def knowledge_points(engine: Engine, course_id=None, scope: CurriculumScope | None = None) -> list[dict]:
    scope = scope or CurriculumScope(course_id=course_id)
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        return [dict(row) for row in conn.execute(text(
            "SELECT k.*, (SELECT count(*) FROM public.edu_chunks ch WHERE ch.knowledge_point_id=k.id AND ch.document_id IS NOT NULL) AS knowledge_chunks, "
            "(SELECT count(*) FROM public.edu_exercise_knowledge_points ek "
            "WHERE ek.knowledge_point_id=k.id) AS exercises "
            "FROM public.edu_knowledge_points k JOIN public.edu_courses c ON c.id=k.course_id "
            "WHERE " + SCOPE_SQL +
            "ORDER BY k.sort_order,k.code"
        ), scope.sql_params()).mappings()]


def knowledge_detail(engine: Engine, knowledge_point_id, scope: CurriculumScope | None = None) -> dict | None:
    params = {"id": str(knowledge_point_id), **(scope or CurriculumScope()).sql_params()}
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        topic = conn.execute(text(
            "SELECT k.* FROM public.edu_knowledge_points k JOIN public.edu_courses c ON c.id=k.course_id "
            "WHERE k.id=CAST(:id AS uuid) AND " + SCOPE_SQL
        ), params).mappings().first()
        if topic is None:
            return None
        cards = [dict(row) for row in conn.execute(text(
            "SELECT c.id,c.content,c.chunk_index,c.source_page,c.metadata,d.title,d.source_uri,d.license_note, "
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
