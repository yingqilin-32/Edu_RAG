"""Import original JSON knowledge cards, then index changed/missing vectors."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, field_validator, model_validator
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from edu_rag.config import ConfigurationError, PROJECT_ROOT, get_settings
from edu_rag.db import get_engine
from edu_rag.embedding import EmbeddingError, count_tokens, get_encoder
from edu_rag.preprocessing import PIPELINE_VERSION, PreprocessingError, TextChunk, clean_text, split_text

DEMO_PATH = PROJECT_ROOT / "data" / "v1" / "math_g3.json"


class Exercise(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    stem: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=2000)
    explanation: str = Field(min_length=1, max_length=4000)
    difficulty: int = Field(default=1, ge=1, le=5)
    question_type: Literal["single_choice", "multiple_choice", "fill_blank", "short_answer", "calculation"] = "calculation"
    options: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("stem", "answer", "explanation", mode="before")
    @classmethod
    def clean_fields(cls, value):
        return clean_text(value).text if isinstance(value, str) else value


class Topic(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    code: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    title: str = Field(min_length=1, max_length=100)
    objective: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=100000)
    boilerplate_lines: list[str] = Field(default_factory=list, max_length=100)
    exercises: list[Exercise] = Field(min_length=1, max_length=100)


class CourseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=200)
    curriculum: str = Field(min_length=1, max_length=200)
    school_stage: Literal["primary", "junior_secondary", "senior_secondary"]
    grade: int = Field(ge=1, le=12)
    subject: Literal["math", "physics", "chemistry", "english"]
    semester: Literal["first", "second", "full_year"] = "full_year"
    textbook_edition: str = Field(min_length=1, max_length=200)
    language: str = "zh-CN"

    @model_validator(mode="after")
    def grade_matches_stage(self):
        low, high = {"primary": (1, 6), "junior_secondary": (7, 9), "senior_secondary": (10, 12)}[self.school_stage]
        if not low <= self.grade <= high:
            raise ValueError("学段与绝对年级不匹配。")
        return self


class ReferenceSource(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    url: HttpUrl
    title: str = Field(min_length=1, max_length=200)
    usage: Literal["topic_reference"] = "topic_reference"


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str = Field(pattern=r"^[a-z0-9-]+$")
    course_code: str = Field(min_length=1, max_length=100)
    license_note: str = Field(min_length=1)
    course: CourseMetadata | None = None
    source_uri: str = Field(default="", max_length=2000)
    reference_sources: list[ReferenceSource] = Field(default_factory=list, max_length=20)
    topics: list[Topic] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_topics(self):
        if len({t.code for t in self.topics}) != len(self.topics):
            raise ValueError("知识点 code 不能重复。")
        return self


def load_dataset(path=DEMO_PATH) -> Dataset:
    path = Path(path)
    dataset = Dataset.model_validate_json(path.read_text(encoding="utf-8-sig"))
    if not dataset.source_uri:
        dataset.source_uri = "data/v1/math_g3.md" if path.resolve() == DEMO_PATH.resolve() else path.name
    return dataset


def stable_id(key):
    return uuid5(NAMESPACE_URL, "https://edu-rag.local/" + key)


def sync_chunks(conn, *, course_id, topic_id, owner, owner_id, chunks, dataset_id, reference_sources=()):
    # owner is an internal constant, never a user-provided SQL identifier.
    if owner not in ("document_id", "exercise_id"):
        raise ValueError("Invalid chunk owner")
    for index, chunk in enumerate(chunks):
        metadata = json.dumps({**chunk.metadata, "dataset": dataset_id,
                               "reference_sources": reference_sources, "content_origin": "authored",
                               "content_sha256": sha256(chunk.content.encode()).hexdigest()}, ensure_ascii=False)
        # The unique owner/index keys keep IDs stable. Metadata-only changes
        # preserve embeddings; changed content invalidates its own vector.
        conn.execute(text(
            f"INSERT INTO public.edu_chunks AS existing(id,course_id,knowledge_point_id,{owner},chunk_index,content,metadata,source_page) "
            "VALUES (:id,:course,:topic,:owner,:index,:content,CAST(:metadata AS jsonb),:page) "
            f"ON CONFLICT ({owner},chunk_index) WHERE {owner} IS NOT NULL DO UPDATE SET "
            "content=excluded.content,metadata=excluded.metadata,source_page=excluded.source_page, "
            "knowledge_point_id=excluded.knowledge_point_id, "
            "embedding=CASE WHEN existing.content=excluded.content THEN existing.embedding ELSE NULL END, "
            "embedding_model=CASE WHEN existing.content=excluded.content THEN existing.embedding_model ELSE NULL END"
        ), {"id": stable_id(f"{dataset_id}/chunk/{owner_id}/{index}"), "course": course_id,
            "topic": topic_id, "owner": owner_id, "index": index, "content": chunk.content,
            "metadata": metadata, "page": chunk.source_page})
    # Only obsolete derived chunks for this imported owner are removed.
    conn.execute(text(f"DELETE FROM public.edu_chunks WHERE {owner}=:owner AND chunk_index>=:count"),
                 {"owner": owner_id, "count": len(chunks)})
    if owner == "document_id":
        conn.execute(text(
            "UPDATE public.edu_documents SET status=CASE WHEN EXISTS "
            "(SELECT 1 FROM public.edu_chunks WHERE document_id=:owner AND embedding IS NULL) "
            "THEN 'parsed' ELSE 'indexed' END WHERE id=:owner"
        ), {"owner": owner_id})


def prepare_dataset(dataset, *, token_counter=count_tokens, max_tokens=None, overlap_tokens=48):
    budget = min(max_tokens or 448, get_settings().embedding_max_length)
    prepared = []
    for topic in dataset.topics:
        cards = split_text(topic.title, topic.content, token_counter, max_tokens=budget,
                           overlap_tokens=min(overlap_tokens, budget // 4), boilerplate_lines=topic.boilerplate_lines)
        exercises = []
        for exercise in topic.exercises:
            content = exercise.stem
            if exercise.options:
                content += "\n" + json.dumps(exercise.options, ensure_ascii=False)
            tokens = token_counter(content)
            if tokens > get_settings().embedding_max_length:
                raise PreprocessingError("单道题目及选项超过模型 token 上限，请先精简题干；题目不会被拆散。")
            exercises.append([TextChunk(content, {"pipeline": PIPELINE_VERSION, "token_count": tokens,
                                                 "kind": "exercise"}, 1)])
        prepared.append((cards, exercises))
    return prepared


def import_dataset(engine, dataset: Dataset, *, prepared=None) -> dict:
    prepared = prepared if prepared is not None else prepare_dataset(dataset)
    references = [reference.model_dump(mode="json") for reference in dataset.reference_sources]
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(7290172027)"))
        if dataset.course is not None:
            conn.execute(text(
                "INSERT INTO public.edu_courses(id,code,title,curriculum,school_stage,grade,subject,semester,textbook_edition,language) "
                "VALUES (:id,:code,:title,:curriculum,:school_stage,:grade,:subject,:semester,:textbook_edition,:language) "
                "ON CONFLICT(code) DO NOTHING"
            ), {"id": stable_id("course/" + dataset.course_code), "code": dataset.course_code, **dataset.course.model_dump()})
        course = conn.execute(text("SELECT * FROM public.edu_courses WHERE code=:code"), {"code": dataset.course_code}).mappings().first()
        if course is None:
            raise ConfigurationError("课程不存在，请在导入 JSON 中提供完整 course 元数据，或先完成 V0 初始化。")
        if dataset.course and any(course[key] != value for key, value in dataset.course.model_dump().items()):
            raise ConfigurationError("导入的课程元数据与数据库不一致；请使用正确的 course_code，不会覆盖已有课程归属。")
        course_id = course["id"]
        for order, (topic, (cards, exercise_chunks)) in enumerate(zip(dataset.topics, prepared, strict=True), 1):
            conn.execute(text(
                "INSERT INTO public.edu_knowledge_points(id,course_id,code,title,learning_objective,sort_order) "
                "VALUES (:id,:course,:code,:title,:objective,:order) ON CONFLICT(course_id,code) DO NOTHING"
            ), {"id": stable_id(f"{dataset.course_code}/{dataset.dataset_id}/topic/{topic.code}"), "course": course_id,
                "code": topic.code, "title": topic.title, "objective": topic.objective, "order": order + 10})
            topic_id = conn.scalar(text("SELECT id FROM public.edu_knowledge_points WHERE course_id=:course AND code=:code"),
                                   {"course": course_id, "code": topic.code})
            source = f"{dataset.source_uri or dataset.dataset_id}#{topic.code}"
            document_id = conn.scalar(text(
                "INSERT INTO public.edu_documents(id,course_id,source_key,title,source_type,source_uri,license_note,status,metadata) "
                "VALUES (:id,:course,:key,:title,'authored',:source,:license,'parsed',CAST(:metadata AS jsonb)) "
                "ON CONFLICT(course_id,source_key) DO UPDATE SET title=excluded.title, "
                "source_uri=excluded.source_uri,license_note=excluded.license_note,metadata=excluded.metadata RETURNING id"
            ), {"id": stable_id(f"{dataset.course_code}/{dataset.dataset_id}/document/{topic.code}"), "course": course_id,
                "key": f"{dataset.dataset_id}/{topic.code}", "title": topic.title + " · 原创知识卡",
                "source": source, "license": dataset.license_note,
                "metadata": json.dumps({"dataset": dataset.dataset_id, "pipeline": PIPELINE_VERSION,
                                        "content_origin": "authored", "reference_sources": references}, ensure_ascii=False)})
            sync_chunks(conn, course_id=course_id, topic_id=topic_id, owner="document_id", owner_id=document_id,
                        chunks=cards, dataset_id=dataset.dataset_id, reference_sources=references)
            for index, (exercise, chunks) in enumerate(zip(topic.exercises, exercise_chunks, strict=True), 1):
                prefix = "V1" if dataset.dataset_id == "v1-original-g3-math" else dataset.dataset_id.upper()
                code = f"{prefix}-{topic.code.upper()}-{index:03}"
                exercise_id = conn.scalar(text(
                    "INSERT INTO public.edu_exercises(id,course_id,code,stem,question_type,difficulty,options,answer,explanation,source_uri) "
                    "VALUES (:id,:course,:code,:stem,:type,:difficulty,CAST(:options AS jsonb),:answer,:explanation,:source) "
                    "ON CONFLICT(course_id,code) DO UPDATE SET stem=excluded.stem,question_type=excluded.question_type, "
                    "difficulty=excluded.difficulty,options=excluded.options,answer=excluded.answer, "
                    "explanation=excluded.explanation,source_uri=excluded.source_uri RETURNING id"
                ), {"id": stable_id(f"{dataset.course_code}/{dataset.dataset_id}/exercise/{code}"), "course": course_id,
                    "code": code, "stem": exercise.stem, "type": exercise.question_type,
                    "difficulty": exercise.difficulty, "options": json.dumps(exercise.options, ensure_ascii=False),
                    "answer": exercise.answer, "explanation": exercise.explanation, "source": source})
                conn.execute(text(
                    "INSERT INTO public.edu_exercise_knowledge_points(exercise_id,knowledge_point_id,course_id) "
                    "VALUES (:exercise,:topic,:course) ON CONFLICT DO NOTHING"
                ), {"exercise": exercise_id, "topic": topic_id, "course": course_id})
                sync_chunks(conn, course_id=course_id, topic_id=topic_id, owner="exercise_id", owner_id=exercise_id,
                            chunks=chunks, dataset_id=dataset.dataset_id, reference_sources=references)
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
    parser = argparse.ArgumentParser(description="V2 清洗、切分、课程校验与真实向量化")
    parser.add_argument("command", choices=("demo", "embed", "import", "preview"))
    parser.add_argument("--file", type=Path, help="结构化知识库 JSON 文件")
    parser.add_argument("--chunk-tokens", type=int, default=448)
    parser.add_argument("--overlap-tokens", type=int, default=48)
    parser.add_argument("--no-embed", action="store_true", help="仅导入样例内容")
    args = parser.parse_args()
    engine = None
    try:
        if args.command in ("import", "preview") and args.file is None:
            parser.error("import / preview 需要 --file。")
        if args.chunk_tokens < 32 or not 0 <= args.overlap_tokens < args.chunk_tokens:
            parser.error("chunk-tokens 至少为 32，overlap-tokens 必须小于 chunk-tokens。")
        dataset = load_dataset(args.file or DEMO_PATH) if args.command != "embed" else None
        prepared = prepare_dataset(dataset, max_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens) if dataset else None
        if args.command == "preview":
            print(json.dumps({"dataset": dataset.dataset_id, "topics": len(dataset.topics),
                              "chunks": [{"content": chunk.content, "source_page": chunk.source_page, **chunk.metadata}
                                         for cards, exercises in prepared for chunk in cards + [c for group in exercises for c in group]]},
                             ensure_ascii=False, indent=2))
            return 0
        engine = get_engine()
        result = import_dataset(engine, dataset, prepared=prepared) if dataset else {}
        if not args.no_embed:
            result.update(embed_pending(engine))
        print(json.dumps({"status": "ok", **result}, ensure_ascii=False, indent=2))
        return 0
    except (ConfigurationError, EmbeddingError, PreprocessingError) as exc:
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
