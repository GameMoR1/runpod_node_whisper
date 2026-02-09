from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import httpx

from app.config import settings
from app.ffmpeg_proc import preprocess_to_wav
from app.gpu import gpu_count, gpu_metrics, gpu_name, torch_cuda_available, torch_cuda_device_count
from app.model_registry import ModelRegistry
from app.types import GpuState, JobRecord
from app.utils_time import now_ms, ms_to_s
from app.whisper_runner import transcribe_on_gpu


logger = logging.getLogger("whisper_node.queue")


class JobQueue:
    def __init__(self, *, model_registry: ModelRegistry) -> None:
        self._models = model_registry
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: dict[str, JobRecord] = {}
        self._gpu_running: dict[int, str] = {}
        self._workers: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    async def start_workers(self) -> None:
        n = gpu_count()
        if n <= 0:
            raise RuntimeError("no NVIDIA GPUs detected")

        if not torch_cuda_available() or torch_cuda_device_count() <= 0:
            raise RuntimeError(
                "torch is installed without CUDA support; install a CUDA-enabled PyTorch build"
            )
        logger.info("starting workers: %d", n)
        self._workers = []
        self._stop.clear()
        for idx in range(n):
            self._workers.append(asyncio.create_task(self._worker_loop(idx)))

    async def stop_workers(self) -> None:
        self._stop.set()
        for _ in self._workers:
            self._q.put_nowait("__stop__")
        for t in self._workers:
            try:
                await t
            except Exception:
                pass
        self._workers = []

    async def enqueue(self, *, job_id: str, model: str, language: str, callback_url: str, file_dir: str) -> None:
        jr = JobRecord(
            job_id=job_id,
            status="queued",
            model=model,
            language=language,
            callback_url=callback_url,
            created_at_ms=now_ms(),
            started_at_ms=None,
            finished_at_ms=None,
            result=None,
            error=None,
            callback_delivered_at_ms=None,
            callback_error=None,
            file_dir=file_dir,
        )
        self._jobs[job_id] = jr
        await self._q.put(job_id)
        logger.info("job queued: %s model=%s language=%s", job_id, model, language)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def snapshot_ids(self) -> tuple[list[str], list[str]]:
        queued = [j.job_id for j in self._jobs.values() if j.status == "queued"]
        running = [j.job_id for j in self._jobs.values() if j.status == "running"]
        return (queued, running)

    def serialize_jobs_public(self) -> dict[str, Any]:
        queued, running = self.snapshot_ids()
        total = len(self._jobs)
        return {
            "total": total,
            "queued": len(queued),
            "running": len(running),
            "queued_ids": queued,
            "running_ids": running,
        }

    def serialize_gpus_public(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        n = gpu_count()
        for i in range(n):
            util, used_mb, total_mb = gpu_metrics(i)
            used_pct = (used_mb / total_mb * 100.0) if total_mb else 0.0
            job_id = self._gpu_running.get(i)
            model = self._jobs[job_id].model if job_id and job_id in self._jobs else None
            items.append(
                {
                    "index": i,
                    "name": gpu_name(i),
                    "util_percent": util,
                    "vram_used_mb": used_mb,
                    "vram_total_mb": total_mb,
                    "vram_used_percent": used_pct,
                    "status": "running" if job_id else "idle",
                    "current_job_id": job_id,
                    "current_model": model,
                }
            )
        return items

    def serialize_job(self, job: JobRecord) -> dict[str, Any]:
        started = job.started_at_ms
        finished = job.finished_at_ms
        queue_time_s = ms_to_s((started or job.created_at_ms) - job.created_at_ms)
        processing_time_s = ms_to_s((finished or (started or job.created_at_ms)) - (started or job.created_at_ms))
        return {
            "job_id": job.job_id,
            "status": job.status,
            "model": job.model,
            "language": job.language,
            "queue_time_s": queue_time_s,
            "processing_time_s": processing_time_s,
            "result": job.result,
            "error": job.error,
            "callback": {
                "delivered": job.callback_delivered_at_ms is not None,
                "delivered_at_ms": job.callback_delivered_at_ms,
                "error": job.callback_error,
            },
        }

    async def _worker_loop(self, gpu_index: int) -> None:
        while not self._stop.is_set():
            job_id = await self._q.get()
            if job_id == "__stop__":
                return
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if not self._models.is_model_known(job.model):
                job.status = "failed"
                job.error = "unknown model"
                continue
            job.status = "running"
            job.started_at_ms = now_ms()
            self._gpu_running[gpu_index] = job_id

            logger.info("job started: %s gpu=%d model=%s", job_id, gpu_index, job.model)

            job_dir = Path(job.file_dir)
            in_path = job_dir / "input"
            wav_path = job_dir / "audio.wav"

            try:
                await preprocess_to_wav(str(in_path), str(wav_path))
                logger.info("job preprocessed: %s", job_id)
                result = await transcribe_on_gpu(
                    gpu_index=gpu_index,
                    wav_path=str(wav_path),
                    model_name=job.model,
                    language=job.language,
                )
                job.result = result
                job.status = "completed"
                job.error = None
                logger.info("job completed: %s", job_id)
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                logger.exception("job failed: %s", job_id)
            finally:
                job.finished_at_ms = now_ms()
                self._gpu_running.pop(gpu_index, None)
                try:
                    if wav_path.exists():
                        wav_path.unlink()
                except Exception:
                    pass

            await self._deliver_callback_and_cleanup(job)

    async def _deliver_callback_and_cleanup(self, job: JobRecord) -> None:
        payload = self.serialize_job(job)
        if job.status == "completed" and job.result is not None:
            payload["result"] = {
                "text": job.result.get("text"),
                "segments": job.result.get("segments"),
                "queue_time_s": payload.get("queue_time_s"),
                "processing_time_s": payload.get("processing_time_s"),
                "gpu": job.result.get("gpu"),
                "token_count": job.result.get("token_count"),
            }
            payload["error"] = None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(job.callback_url, json=payload)
                if r.status_code < 200 or r.status_code >= 300:
                    raise RuntimeError("callback failed")
            job.callback_delivered_at_ms = now_ms()
            job.callback_error = None
            logger.info("callback delivered: %s", job.job_id)
        except Exception as e:
            job.callback_error = str(e)
            logger.warning("callback failed: %s", job.job_id)
            return

        try:
            shutil.rmtree(job.file_dir, ignore_errors=True)
        except Exception:
            pass
        logger.info("job cleaned up: %s", job.job_id)
