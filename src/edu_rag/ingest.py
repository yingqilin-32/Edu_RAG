"""Import original JSON knowledge cards, then index changed/missing vectors."""

import argparse
from hashlib import sha256
import json
import sys
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from edu_rag.config import ConfigurationError, PROJECT_ROOT
from edu_rag.db import get_engine
from edu_rag.embedding import EmbeddingError, get_encoder

DEMO_PATH = PROJECT_ROOT / "data" / "v1" / "math_g3.json"


class Exercise(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    stem: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=2000)
    explanation: str = Field(min_length=1, max_length=4000)
    difficulty: int = Field(default=1, ge=1, le=5)
    question_type: Literal["single_choice", "multiple_choice", "fill_blank", "short_answer", "calculation"] = "calculation"
    options: list[dict[str, str]] = Field(default_factory=list)


class Topic(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    code: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    title: str = Field(min_length=1, max_length=100)
    objective: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=1800)
    exercises: list[Exercise] = Field(min_length=1, max_length=100)


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(pattern=r"^[a-z0-9-]+$")
    course_code: str = Field(min_length=1, max_length=100)
    license_note: str = Field(min_length=1)
    topics: list[Topic] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_topics(self):
        if len({t.code for t in self.topics}) != len(self.topics):
            raise ValueError("知识点 code 不能重复。")
        return self


def load_dataset(path=DEMO_PATH) -> Dataset:
    return Dataset.model_validate_json(path.read_text(encoding="utf-8"))


def stable_id(key):
    return uuid5(NAMESPACE_URL, "https://edu-rag.local/" + key)


def upsert_chunk(conn, *, course_id, topic_id, owner, owner_id, content, dataset_id):
    # owner is an internal constant, never a user-provided SQL identifier.
    if owner not in ("document_id", "exercise_id"):
        raise ValueError("Invalid chunk owner")
    metadata = json.dumps({"dataset": dataset_id, "content_sha256": sha256(content.encode()).hexdigest()})
    current = conn.execute(text(
        f"SELECT id,content FROM public.edu_chunks WHERE {owner}=:owner AND chunk_index=0"
    ), {"owner": owner_id}).mappings().first()
    if current:
        if current["content"] != content:
            conn.execute(text(
                "UPDATE public.edu_chunks SET content=:content,metadata=CAST(:metadata AS jsonb), "
                "embedding=NULL,embedding_model=NULL WHERE id=:id"
            ), {"id": current["id"], "content": content, "metadata": metadata})
        return
    conn.execute(text(
        f"INSERT INTO public.edu_chunks(id,course_id,knowledge_point_id,{owner},chunk_index,content,metadata) "
        "VALUES (:id,:course,:topic,:owner,0,:content,CAST(:metadata AS jsonb))"
    ), {"id": stable_id(f"{dataset_id}/chunk/{owner_id}"), "course": course_id,
        "topic": topic_id, "owner": owner_id, "content": content, "metadata": metadata})


def import_dataset(engine, dataset: Dataset) -> dict:
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(7290172027)"))
        course_id = conn.scalar(text("SELECT id FROM public.edu_courses WHERE code=:code"), {"code": dataset.course_code})
        if course_id is None:
            raise ConfigurationError("样例所属课程不存在，请先完成 V0 migrate 和 seed。")
        for order, topic in enumerate(dataset.topics, 1):
            conn.execute(text(
                "INSERT INTO public.edu_knowledge_points(id,course_id,code,title,learning_objective,sort_order) "
                "VALUES (:id,:course,:code,:title,:objective,:order) ON CONFLICT(course_id,code) DO NOTHING"
            ), {"id": stable_id(f"{dataset.dataset_id}/topic/{topic.code}"), "course": course_id,
                "code": topic.code, "title": topic.title, "objective": topic.objective, "order": order + 10})
            topic_id = conn.scalar(text("SELECT id FROM public.edu_knowledge_points WHERE course_id=:course AND code=:code"),
                                   {"course": course_id, "code": topic.code})
            source = f"data/v1/math_g3.md#{topic.code}"
            document_id = conn.scalar(text(
                "INSERT INTO public.edu_documents(id,course_id,source_key,title,source_type,source_uri,license_note,status,metadata) "
                "VALUES (:id,:course,:key,:title,'authored',:source,:license,'parsed',CAST(:metadata AS jsonb)) "
                "ON CONFLICT(course_id,source_key) DO UPDATE SET title=excluded.title, "
                "source_uri=excluded.source_uri,license_note=excluded.license_note RETURNING id"
            ), {"id": stable_id(f"{dataset.dataset_id}/document/{topic.code}"), "course": course_id,
                "key": f"{dataset.dataset_id}/{topic.code}", "title": topic.title + " · 原创知识卡",
                "source": source, "license": dataset.license_note,
                "metadata": json.dumps({"dataset": dataset.dataset_id})})
            upsert_chunk(conn, course_id=course_id, topic_id=topic_id, owner="document_id", owner_id=document_id,
                         content=f"# {topic.title}\n\n{topic.content}", dataset_id=dataset.dataset_id)
            for index, exercise in enumerate(topic.exercises, 1):
                code = f"V1-{topic.code.upper()}-{index:03}"
                exercise_id = conn.scalar(text(
                    "INSERT INTO public.edu_exercises(id,course_id,code,stem,question_type,difficulty,options,answer,explanation,source_uri) "
                    "VALUES (:id,:course,:code,:stem,:type,:difficulty,CAST(:options AS jsonb),:answer,:explanation,:source) "
                    "ON CONFLICT(course_id,code) DO UPDATE SET stem=excluded.stem,question_type=excluded.question_type, "
                    "difficulty=excluded.difficulty,options=excluded.options,answer=excluded.answer, "
                    "explanation=excluded.explanation,source_uri=excluded.source_uri RETURNING id"
                ), {"id": stable_id(f"{dataset.dataset_id}/exercise/{code}"), "course": course_id,
                    "code": code, "stem": exercise.stem, "type": exercise.question_type,
                    "difficulty": exercise.difficulty, "options": json.dumps(exercise.options, ensure_ascii=False),
                    "answer": exercise.answer, "explanation": exercise.explanation, "source": source})
                conn.execute(text(
                    "INSERT INTO public.edu_exercise_knowledge_points(exercise_id,knowledge_point_id,course_id) "
                    "VALUES (:exercise,:topic,:course) ON CONFLICT DO NOTHING"
                ), {"exercise": exercise_id, "topic": topic_id, "course": course_id})
                retrieval_text = exercise.stem
                if exercise.options:
                    retrieval_text += "\n" + json.dumps(exercise.options, ensure_ascii=False)
                upsert_chunk(conn, course_id=course_id, topic_id=topic_id, owner="exercise_id", owner_id=exercise_id,
                             content=retrieval_text, dataset_id=dataset.dataset_id)
    return {"dataset": dataset.dataset_id, "topics_in_dataset": len(dataset.topics),
            "exercises_in_dataset": sum(len(t.exercises) for t in dataset.topics)}


def embed_pending(engine, encoder=None) -> dict:
    encoder = encoder if encoder is not None else get_encoder()
    with engine.connect() as conn:
        pending = [dict(row) for row in conn.execute(text(
            "SELECT id,content FROM public.edu_chunks WHERE embedding IS NULL OR embedding_model IS DISTINCT FROM :model ORDER BY id"
        ), {"model": encoder.model_id}).mappings()]
    embedded = 0
    for offset in range(0, len(pending), 8):
        batch = pending[offset:offset + 8]
        vectors = encoder.encode([row["content"] for row in batch])
        with engine.begin() as conn:
            for row, vector in zip(batch, vectors, strict=True):
                result = conn.execute(text(
                    "UPDATE public.edu_chunks SET embedding=CAST(:vector AS extensions.vector),embedding_model=:model "
                    "WHERE id=:id AND content=:content"
                ), {**row, "vector": json.dumps(vector, allow_nan=False), "model": encoder.model_id})
                embedded += result.rowcount
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE public.edu_documents d SET status=CASE WHEN EXISTS "
            "(SELECT 1 FROM public.edu_chunks c WHERE c.document_id=d.id) AND NOT EXISTS "
            "(SELECT 1 FROM public.edu_chunks c WHERE c.document_id=d.id AND "
            "(c.embedding IS NULL OR c.embedding_model IS DISTINCT FROM :model)) THEN 'indexed' ELSE 'parsed' END"
        ), {"model": encoder.model_id})
    return {"embedded_this_run": embedded, "embedding_model": encoder.model_id}


def main():
    parser = argparse.ArgumentParser(description="V1 原创知识库导入与真实向量化")
    parser.add_argument("command", choices=("demo", "embed"))
    parser.add_argument("--no-embed", action="store_true", help="仅导入样例内容")
    args = parser.parse_args()
    engine = None
    try:
        engine = get_engine()
        result = import_dataset(engine, load_dataset()) if args.command == "demo" else {}
        if not args.no_embed:
            result.update(embed_pending(engine))
        print(json.dumps({"status": "ok", **result}, ensure_ascii=False, indent=2))
        return 0
    except (ConfigurationError, EmbeddingError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValidationError):
        print("样例文件缺失或格式不合法，请检查 data/v1/math_g3.json。", file=sys.stderr)
        return 2
    except SQLAlchemyError:
        print("数据库导入失败，请检查连接及 V0 schema。", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
