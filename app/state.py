from __future__ import annotations

import os
import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import settings
from app.db import dispose_engine, fetch_hugging_face_token
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
        if self.models is None or self.queue is None:
            self.health_status = "error"
            self.health_error = "service not initialized"
            return

        logger.info("initializing service")
        try:
            _check_ffmpeg()
            _check_whisper_module()
        except Exception as e:
            self.health_status = "error"
            self.health_error = str(e)
            logger.exception("service failed to initialize")
            return

        try:
            token = await fetch_hugging_face_token()
            if token:
                os.environ["HF_TOKEN"] = token
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        except Exception:
            pass

        while True:
            try:
                await self.models.load_from_db_and_prepare()
                details = self.models.unready_details()
                if details:
                    raise RuntimeError(f"model preparation failed: {details}")

                await self.queue.start_workers()
                self.health_status = "ready"
                self.health_error = None
                logger.info("service ready")
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.health_status = "error"
                self.health_error = str(e)
                logger.warning("service not ready: %s", e)
                await asyncio.sleep(max(5, int(settings.MODEL_PREPARE_RETRY_S)))

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
