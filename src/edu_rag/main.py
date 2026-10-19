"""Local V3 knowledge browser and scoped hybrid RAG API."""

from pathlib import Path
from uuid import UUID
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from edu_rag import __version__
from edu_rag.config import ConfigurationError
from edu_rag.db import database_status, get_engine
from edu_rag import knowledge, rag, workflow, deployment, adaptive
from edu_rag.embedding import EmbeddingError
from edu_rag.scope import CurriculumScope
from edu_rag import conversation

app = FastAPI(title="Edu RAG", version=__version__)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "stage": "V6"}


@app.get("/health/db")
def health_db() -> dict:
    try:
        return database_status(get_engine())
    except (ConfigurationError, SQLAlchemyError):
        raise HTTPException(
            status_code=503,
            detail="数据库尚未就绪。请在本机完成 .env 配置、migrate，再运行 db check。",
        ) from None


@app.get("/api/v0/courses")
def list_courses() -> list[dict]:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            rows = conn.execute(text(
                "SELECT id, code, title, curriculum, school_stage, grade, subject, "
                "semester, textbook_edition FROM public.edu_courses ORDER BY code"
            ))
            return [dict(row) for row in rows.mappings()]
    except (ConfigurationError, SQLAlchemyError):
        raise HTTPException(status_code=503, detail="数据库尚未就绪，请先完成 V0 初始化。") from None


@app.get("/api/v1/courses")
@app.get("/api/v2/courses")
@app.get("/api/v3/courses")
def v1_courses(scope: Annotated[CurriculumScope, Query()]):
    try:
        return knowledge.courses(get_engine(), scope)
    except (ConfigurationError, SQLAlchemyError):
        raise HTTPException(503, "数据库尚未就绪，请检查本机连接。") from None


@app.get("/api/v1/knowledge-points")
@app.get("/api/v2/knowledge-points")
@app.get("/api/v3/knowledge-points")
def topics(scope: Annotated[CurriculumScope, Query()]):
    try:
        return knowledge.knowledge_points(get_engine(), scope=scope)
    except (ConfigurationError, SQLAlchemyError):
        raise HTTPException(503, "暂时无法读取知识目录。") from None


@app.get("/api/v1/knowledge-points/{knowledge_point_id}")
@app.get("/api/v2/knowledge-points/{knowledge_point_id}")
@app.get("/api/v3/knowledge-points/{knowledge_point_id}")
def topic_detail(knowledge_point_id: UUID, scope: Annotated[CurriculumScope, Query()]):
    try:
        result = knowledge.knowledge_detail(get_engine(), knowledge_point_id, scope)
    except (ConfigurationError, SQLAlchemyError):
        raise HTTPException(503, "暂时无法读取知识正文。") from None
    if result is None:
        raise HTTPException(404, "知识点不存在。")
    return result


def run_rag(operation, request):
    try:
        return operation(request)
    except (ConfigurationError, EmbeddingError) as exc:
        raise HTTPException(503, str(exc)) from None
    except rag.ModelError as exc:
        raise HTTPException(502, str(exc)) from None
    except SQLAlchemyError:
        raise HTTPException(503, "检索数据库暂时不可用。") from None


@app.post("/api/v1/search")
@app.post("/api/v2/search")
@app.post("/api/v3/search")
def search(request: rag.SearchRequest):
    return run_rag(rag.search, request)


@app.post("/api/v3/plan")
def plan(request: rag.SearchRequest):
    return run_rag(rag.construct_plan, request)


@app.post("/api/v1/ask")
@app.post("/api/v2/ask")
@app.post("/api/v3/ask")
def ask(request: rag.SearchRequest):
    return run_rag(rag.ask, request)


@app.post("/api/v4/ask")
def v4_ask(request: workflow.WorkflowRequest):
    return run_rag(deployment.ask, request)


@app.post("/api/v4/search")
def v4_search(request: workflow.WorkflowRequest):
    return run_rag(deployment.search, request)


@app.post("/api/v5/ask")
def v5_ask(request: adaptive.AdaptiveRequest):
    return run_rag(adaptive.ask, request)


@app.post("/api/v5/search")
def v5_search(request: adaptive.AdaptiveRequest):
    return run_rag(adaptive.search, request)


@app.post("/api/v6/ask")
def v6_ask(request: conversation.ConversationRequest):
    try:
        return run_rag(conversation.service.run, request)
    except conversation.SessionError as exc:
        raise HTTPException(exc.status, str(exc)) from None


@app.delete("/api/v6/sessions/{session_id}")
def v6_delete(session_id: UUID):
    try:
        conversation.service.delete(session_id)
        return {"status": "cleared"}
    except conversation.SessionError as exc:
        raise HTTPException(exc.status, str(exc)) from None


@app.post("/api/v6/search")
def v6_search(request: conversation.ConversationRequest):
    try:
        return run_rag(lambda req: conversation.service.run(req, preview=True), request)
    except conversation.SessionError as exc:
        raise HTTPException(exc.status, str(exc)) from None

