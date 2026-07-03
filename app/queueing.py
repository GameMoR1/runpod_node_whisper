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
from app.error_log import log_node_error
from app.channel_proc import (
    merge_channel_results,
    prepare_channel_wavs,
    transcribe_channel_wav,
)
from app.gpu import gpu_count, gpu_metrics, gpu_name, torch_cuda_available, torch_cuda_device_count
from app.model_registry import ModelRegistry
from app.types import GpuState, JobRecord
from app.utils_time import now_ms, ms_to_s
from app.vad_chunking import pipeline_info
from app.whisper_runner import transcribe_chunks_on_gpu


logger = logging.getLogger("whisper_node.queue")


class JobQueue:
    def __init__(self, *, model_registry: ModelRegistry) -> None:
        self._models = model_registry
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: dict[str, JobRecord] = {}
        self._gpu_running: dict[int, str] = {}
        self._workers: list[asyncio.Task] = []
        self._stop = asyncio.Event()
        self._failed_total = 0
        self._completed_total = 0

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

    async def enqueue(
        self,
        *,
        job_id: str,
        model: str,
        language: str,
        callback_url: str,
        file_dir: str,
        split_by_channels: bool = False,
    ) -> None:
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
            split_by_channels=split_by_channels,
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
        completed = [j.job_id for j in self._jobs.values() if j.status == "completed"]
        failed = [j.job_id for j in self._jobs.values() if j.status == "failed"]
        return {
            "total": len(completed),
            "queued": len(queued),
            "running": len(running),
            "failed": len(failed),
            "queued_ids": queued,
            "running_ids": running,
            "failed_ids": failed,
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
                await log_node_error(
                    component="queue",
                    message=job.error,
                    job_id=job_id,
                    context={"model": job.model},
                )
                continue
            job.status = "running"
            job.started_at_ms = now_ms()
            self._gpu_running[gpu_index] = job_id

            logger.info("job started: %s gpu=%d model=%s", job_id, gpu_index, job.model)

            job_dir = Path(job.file_dir)
            in_path = job_dir / "input"
            wav_paths: list[Path] = []

            try:
                channel_sources, preprocess_meta = await prepare_channel_wavs(
                    str(in_path),
                    job_dir,
                    split_by_channels=job.split_by_channels,
                )
                wav_paths = [path for _ch, path in channel_sources]
                logger.info(
                    "job preprocessed: %s mode=%s channels=%s",
                    job_id,
                    preprocess_meta.get("mode"),
                    preprocess_meta.get("source_channels"),
                )

                split_active = bool(
                    job.split_by_channels and preprocess_meta.get("source_channels", 1) > 1
                )
                channel_results: list[tuple[int, dict[str, Any]]] = []
                total_segments = 0
                total_chunks = 0

                for ch_idx, wav_path in channel_sources:
                    ch_dir = job_dir / f"ch{ch_idx}"
                    ch_dir.mkdir(parents=True, exist_ok=True)
                    ch_result = await transcribe_channel_wav(
                        gpu_index=gpu_index,
                        wav_path=wav_path,
                        job_dir=ch_dir,
                        model_name=job.model,
                        language=job.language,
                        transcribe_chunks_fn=transcribe_chunks_on_gpu,
                        channel_index=ch_idx,
                        tag_speaker=split_active or job.split_by_channels,
                    )
                    channel_results.append((ch_idx, ch_result))
                    total_segments += len(ch_result.get("segments") or [])

                if split_active and len(channel_results) > 1:
                    result = merge_channel_results(channel_results)
                else:
                    result = channel_results[0][1]
                    if job.split_by_channels:
                        speaker = "speaker_0"
                        result["speakers"] = [{"id": speaker, "channel": 0}]
                        for seg in result.get("segments") or []:
                            if isinstance(seg, dict) and "speaker" not in seg:
                                seg["speaker"] = speaker
                                seg["channel"] = 0

                if isinstance(result, dict):
                    result["pipeline"] = {
                        "preprocess": preprocess_meta,
                        "vad": {"method": settings.VAD_METHOD},
                        "chunking": {"mode": "vad_hybrid", "segments": total_segments, "chunks": total_chunks},
                        "split_by_channels": job.split_by_channels,
                    }
                job.result = result
                job.status = "completed"
                job.error = None
                self._completed_total += 1
                logger.info("job completed: %s", job_id)
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                self._failed_total += 1
                logger.exception("job failed: %s", job_id)
                await log_node_error(
                    component="queue",
                    message=job.error,
                    job_id=job_id,
                    context={"model": job.model, "gpu_index": gpu_index},
                    exc=e,
                )
            finally:
                job.finished_at_ms = now_ms()
                self._gpu_running.pop(gpu_index, None)
                for wav_path in wav_paths:
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
                "speakers": job.result.get("speakers"),
                "queue_time_s": payload.get("queue_time_s"),
                "processing_time_s": payload.get("processing_time_s"),
                "gpu": job.result.get("gpu"),
                "token_count": job.result.get("token_count"),
                "vram_peak_allocated_mb": job.result.get("vram_peak_allocated_mb"),
                "language": job.language,
            }
            payload["error"] = None

        timeout = float(settings.CALLBACK_TIMEOUT_S)
        timeout = float(settings.CALLBACK_TIMEOUT_S)
        attempts = max(1, int(settings.CALLBACK_ATTEMPTS))
        for attempt in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.post(job.callback_url, json=payload)
                    if r.status_code < 200 or r.status_code >= 300:
                        raise RuntimeError(f"callback HTTP {r.status_code}")
                job.callback_delivered_at_ms = now_ms()
                job.callback_error = None
                logger.info("callback delivered: %s", job.job_id)
                break
            except Exception as e:
                if attempt + 1 < attempts:
                    wait_s = min(30, 2**attempt)
                    logger.warning(
                        "callback failed: %s (attempt %d/%d, retry in %ss)",
                        job.job_id,
                        attempt + 1,
                        attempts,
                        wait_s,
                    )
                    await asyncio.sleep(wait_s)
                    continue
                job.callback_error = str(e)
                logger.warning("callback failed: %s", job.job_id)
                await log_node_error(
                    component="callback",
                    message=str(e),
                    job_id=job.job_id,
                    context={"callback_url": job.callback_url, "status": job.status},
                    exc=e,
                )
                return

        try:
            shutil.rmtree(job.file_dir, ignore_errors=True)
        except Exception:
            pass
        logger.info("job cleaned up: %s", job.job_id)
