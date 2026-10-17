"""V0 read-only API. Start locally; database writes are explicit CLI commands."""

from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from edu_rag import __version__
from edu_rag.config import ConfigurationError
from edu_rag.db import database_status, get_engine

app = FastAPI(title="Edu RAG", version=__version__)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "stage": "V0"}


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

