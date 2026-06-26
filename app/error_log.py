from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Optional

from sqlalchemy import text

from app.db import get_engine

logger = logging.getLogger(__name__)

ERROR_LOG_TABLE = "whisper_node_error_log"


async def log_node_error(
    *,
    component: str,
    message: str,
    level: str = "error",
    error_type: Optional[str] = None,
    job_id: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    stack_trace: Optional[str] = None,
    exc: Optional[BaseException] = None,
) -> None:
    if exc is not None:
        if error_type is None:
            error_type = type(exc).__name__
        if stack_trace is None:
            stack_trace = traceback.format_exc()
        if not message:
            message = str(exc)

    ctx_text = json.dumps(context, ensure_ascii=False, default=str) if context else None
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {ERROR_LOG_TABLE} (
                        level, component, message, error_type,
                        job_id, context, stack_trace
                    )
                    VALUES (
                        :level, :component, :message, :error_type,
                        :job_id, CAST(:context AS jsonb), :stack_trace
                    )
                    """
                ),
                {
                    "level": level,
                    "component": component,
                    "message": message,
                    "error_type": error_type,
                    "job_id": job_id,
                    "context": ctx_text,
                    "stack_trace": stack_trace,
                },
            )
    except Exception:
        logger.debug("failed to persist node error log", exc_info=True)
