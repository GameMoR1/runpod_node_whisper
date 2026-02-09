from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote_plus

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import settings


def _build_dsn() -> str:
    user = quote_plus(settings.DB_USER)
    pw = quote_plus(settings.DB_PASS)
    host = settings.DB_HOST
    port = settings.DB_PORT
    name = settings.DB_NAME
    return f"postgresql+psycopg://{user}:{pw}@{host}:{port}/{name}"


_engine: Optional[AsyncEngine] = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is not None:
        return _engine
    try:
        _engine = create_async_engine(
            _build_dsn(),
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "database driver is missing; install dependencies from requirements.txt"
        ) from e
    return _engine


async def dispose_engine() -> None:
    global _engine
    if _engine is None:
        return
    await _engine.dispose()
    _engine = None


async def fetch_all(sql: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    engine = get_engine()
    async with engine.connect() as conn:
        res = await conn.execute(text(sql), params or {})
        rows = res.mappings().all()
        return [dict(r) for r in rows]


async def fetch_one(sql: str, params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    rows = await fetch_all(sql, params)
    return rows[0] if rows else None
