from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from app.config import settings
from app.db import fetch_all
from app.types import ModelState


logger = logging.getLogger("whisper_node.models")


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelState] = {}
        self._lock = asyncio.Lock()

    def is_model_known(self, model_name: str) -> bool:
        return model_name in self._models

    def all(self) -> list[ModelState]:
        return list(self._models.values())

    def serialize_public(self) -> list[dict]:
        items = []
        for m in self.all():
            items.append(
                {
                    "id_model": m.id_model,
                    "model_name": m.model_name,
                    "enabled": m.enabled,
                    "status": m.status,
                    "progress": m.progress,
                }
            )
        return items

    async def load_from_db_and_prepare(self) -> None:
        logger.info("loading models from db")
        try:
            rows = await fetch_all(
                """
                SELECT id_model, model_name
                FROM whisper_models
                WHERE source = :source
                """,
                {"source": "whisper"},
            )
        except Exception:
            rows = await fetch_all(
                """
                SELECT model_id AS id_model, model_name
                FROM whisper_models
                WHERE source = :source
                """,
                {"source": "whisper"},
            )

        try:
            enabled_ids = await fetch_all(
                """
                SELECT DISTINCT model_id
                FROM model_settings
                WHERE enabled = :enabled
                """,
                {"enabled": True},
            )
        except Exception:
            enabled_ids = await fetch_all(
                """
                SELECT DISTINCT id_model AS model_id
                FROM model_settings
                WHERE enabled = :enabled
                """,
                {"enabled": True},
            )
        enabled_set = {int(r["model_id"]) for r in enabled_ids if r.get("model_id") is not None}

        async with self._lock:
            self._models = {}
            for r in rows:
                mid = int(r["id_model"])
                name = str(r["model_name"])
                enabled = mid in enabled_set
                if not enabled:
                    continue
                self._models[name] = ModelState(
                    id_model=mid,
                    model_name=name,
                    enabled=True,
                    status="queued_for_download",
                    progress=0.0,
                    error=None,
                )

        if not self._models:
            raise RuntimeError("no enabled whisper models")

        logger.info("enabled whisper models: %d", len(self._models))

        for name in list(self._models.keys()):
            await self._download_model(name)

        async with self._lock:
            bad = [m for m in self._models.values() if m.status != "downloaded"]
            if bad:
                details = ", ".join(
                    f"{m.model_name}(status={m.status}, error={m.error})" for m in bad
                )
                raise RuntimeError(f"model preparation failed: {details}")

    async def _download_model(self, model_name: str) -> None:
        async with self._lock:
            st = self._models.get(model_name)
            if st is None:
                return
            st.status = "downloading"
            st.progress = 0.0
            st.error = None

        logger.info("downloading model: %s", model_name)

        def _blocking_download() -> None:
            Path(settings.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
            import whisper

            m = whisper.load_model(model_name, device="cpu", download_root=settings.MODEL_CACHE_DIR)
            del m

        try:
            await asyncio.to_thread(_blocking_download)
            async with self._lock:
                st2 = self._models.get(model_name)
                if st2 is not None:
                    st2.status = "downloaded"
                    st2.progress = 100.0
            logger.info("model downloaded: %s", model_name)
        except Exception as e:
            async with self._lock:
                st2 = self._models.get(model_name)
                if st2 is not None:
                    st2.status = "error"
                    st2.error = str(e)
            logger.exception("model download failed: %s", model_name)
