from __future__ import annotations

import os
import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import settings
from app.db import dispose_engine
from app.model_registry import ModelRegistry
from app.queueing import JobQueue
from app.types import HealthStatus


logger = logging.getLogger("whisper_node")


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")


def _check_whisper_module() -> None:
    try:
        import whisper  # noqa: F401
    except Exception as e:
        raise RuntimeError("python module 'whisper' is not installed") from e


@dataclass
class AppState:
    health_status: HealthStatus = "starting"
    health_error: Optional[str] = None

    models: Optional[ModelRegistry] = None
    queue: Optional[JobQueue] = None

    _init_task: Optional[asyncio.Task] = None

    async def startup(self) -> None:
        Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

        self.models = ModelRegistry()
        self.queue = JobQueue(model_registry=self.models)
        self.health_status = "starting"
        self.health_error = None
        self._init_task = asyncio.create_task(self._initialize())

    async def _initialize(self) -> None:
        try:
            if self.models is None or self.queue is None:
                raise RuntimeError("service not initialized")
            logger.info("initializing service")
            _check_ffmpeg()
            _check_whisper_module()
            await self.models.load_from_db_and_prepare()
            await self.queue.start_workers()
            self.health_status = "ready"
            self.health_error = None
            logger.info("service ready")
        except Exception as e:
            self.health_status = "error"
            self.health_error = str(e)
            logger.exception("service failed to initialize")

    async def shutdown(self) -> None:
        if self._init_task is not None:
            self._init_task.cancel()
            try:
                await self._init_task
            except BaseException:
                pass
        if self.queue is not None:
            await self.queue.stop_workers()
        await dispose_engine()
