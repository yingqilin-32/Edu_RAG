"""Versioned migrations, optional demo data, and read-only database checks.

Run from the project checkout: python -m edu_rag.db migrate|seed|check
"""

import argparse
from functools import lru_cache
import json
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from edu_rag.config import ConfigurationError, PROJECT_ROOT, get_settings

TABLES = (
    "edu_courses",
    "edu_knowledge_points",
    "edu_documents",
    "edu_exercises",
    "edu_exercise_knowledge_points",
    "edu_chunks",
)


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url(),
        connect_args={
            "sslmode": settings.db_sslmode,
            "connect_timeout": settings.db_connect_timeout,
            "prepare_threshold": None,
        },
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=0,
        hide_parameters=True,
    )


def migrate(engine: Engine) -> list[str]:
    """Apply each new migration once; all pending files share one transaction."""
    applied = []
    migrations = sorted((PROJECT_ROOT / "supabase" / "migrations").glob("*.sql"))
    if not migrations:
        raise ConfigurationError("找不到迁移文件；请在完整项目中以 editable 方式安装。")
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_xact_lock(7290172026)"))
        has_history = conn.scalar(text("SELECT to_regclass('public.edu_schema_migrations')"))
        versions = set()
        if has_history:
            versions = set(conn.scalars(text("SELECT version FROM public.edu_schema_migrations")))
        for path in migrations:
            version = path.name.split("_", 1)[0]
            if version in versions:
                continue
            conn.exec_driver_sql(path.read_text(encoding="utf-8"))
            applied.append(path.name)
    return applied


def seed(engine: Engine) -> None:
    """Insert only authored demo records; existing records are never overwritten."""
    with engine.begin() as conn:
        conn.exec_driver_sql((PROJECT_ROOT / "supabase" / "seed.sql").read_text(encoding="utf-8"))


def database_status(engine: Engine) -> dict:
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        vector_version = conn.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
        counts = {name: conn.scalar(text(f"SELECT count(*) FROM public.{name}")) for name in TABLES}
        embedded = conn.scalar(text("SELECT count(*) FROM public.edu_chunks WHERE embedding IS NOT NULL"))
        return {
            "status": "ok",
            "pgvector_version": vector_version,
            "tables": counts,
            "embedded_chunks": embedded,
            "embedding_dimensions": 1024,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Edu RAG V0 database tools")
    parser.add_argument("command", choices=("migrate", "seed", "check"))
    args = parser.parse_args()
    try:
        engine = get_engine()
        if args.command == "migrate":
            result = {"status": "ok", "applied": migrate(engine)}
        elif args.command == "seed":
            seed(engine)
            result = {"status": "ok", "message": "演示数据已导入；已有记录保持原样。"}
        else:
            result = database_status(engine)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except SQLAlchemyError as exc:
        state = getattr(getattr(exc, "orig", None), "sqlstate", None) or "unavailable"
        print(
            f"数据库操作失败（SQLSTATE: {state}）。检查 .env、Supabase 项目状态和迁移顺序。"
            " 请勿把数据库密码或完整连接字符串发到聊天中。",
            file=sys.stderr,
        )
        return 1
    finally:
        if get_engine.cache_info().currsize:
            get_engine().dispose()


if __name__ == "__main__":
    raise SystemExit(main())

