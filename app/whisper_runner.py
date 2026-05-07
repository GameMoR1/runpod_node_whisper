from __future__ import annotations

import asyncio
from typing import Any

from app.config import settings
from app.gpu import gpu_metrics
from app.postprocess import postprocess_text
from app.vad_chunking import AudioChunk


async def transcribe_on_gpu(
    *,
    gpu_index: int,
    wav_path: str,
    model_name: str,
    language: str,
) -> dict[str, Any]:
    util_samples: list[float] = []
    vram_samples: list[float] = []
    vram_total_mb: float = 0.0

    stop = asyncio.Event()

    async def sampler() -> None:
        nonlocal vram_total_mb
        while not stop.is_set():
            util, used_mb, total_mb = gpu_metrics(gpu_index)
            util_samples.append(util)
            vram_samples.append(used_mb)
            if total_mb:
                vram_total_mb = total_mb
            await asyncio.sleep(0.5)

    def run_blocking_sync() -> dict[str, Any]:
        import torch
        import whisper

        torch.cuda.set_device(gpu_index)
        torch.cuda.reset_peak_memory_stats(gpu_index)
        model = whisper.load_model(model_name, device=f"cuda:{gpu_index}", download_root=settings.MODEL_CACHE_DIR)
        try:
            result = model.transcribe(
                wav_path,
                temperature=settings.WHISPER_TEMPERATURE,
                logprob_threshold=settings.WHISPER_LOGPROB_THRESHOLD,
                language=language,
            )
        finally:
            del model
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        text = str(result.get("text") or "")
        text_pp = postprocess_text(text)
        segments = result.get("segments") or []

        token_count = 0
        if text_pp:
            lang_code = language
            try:
                to_code = getattr(whisper.tokenizer, "TO_LANGUAGE_CODE", None)
                if isinstance(to_code, dict):
                    lang_code = to_code.get(str(language).lower(), language)
            except Exception:
                pass
            try:
                tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=lang_code, task="transcribe")
                token_count = len(tokenizer.encode(text_pp))
            except Exception:
                token_count = 0

        peak_alloc_mb = float(torch.cuda.max_memory_allocated(gpu_index)) / (1024 * 1024)
        return {
            "text": text_pp,
            "segments": segments,
            "token_count": token_count,
            "vram_peak_allocated_mb": peak_alloc_mb,
        }

    sampler_task = asyncio.create_task(sampler())
    try:
        res = await asyncio.to_thread(run_blocking_sync)
    finally:
        stop.set()
        try:
            await sampler_task
        except Exception:
            pass

    util, used_mb, total_mb = gpu_metrics(gpu_index)
    if total_mb and not vram_total_mb:
        vram_total_mb = total_mb

    util_avg = sum(util_samples) / len(util_samples) if util_samples else util
    util_max = max(util_samples) if util_samples else util
    vram_used_avg = sum(vram_samples) / len(vram_samples) if vram_samples else used_mb
    vram_used_max = max(vram_samples) if vram_samples else used_mb
    vram_used_pct_max = (vram_used_max / vram_total_mb * 100.0) if vram_total_mb else 0.0
    vram_used_pct = (used_mb / vram_total_mb * 100.0) if vram_total_mb else 0.0

    res["gpu"] = {
        "index": gpu_index,
        "util_avg_percent": util_avg,
        "util_max_percent": util_max,
        "vram_total_mb": vram_total_mb,
        "vram_used_avg_mb": vram_used_avg,
        "vram_used_max_mb": vram_used_max,
        "vram_used_percent": vram_used_pct,
        "vram_used_percent_max": vram_used_pct_max,
    }
    return res


async def transcribe_chunks_on_gpu(
    *,
    gpu_index: int,
    chunks: list[AudioChunk],
    model_name: str,
    language: str,
) -> dict[str, Any]:
    util_samples: list[float] = []
    vram_samples: list[float] = []
    vram_total_mb: float = 0.0

    stop = asyncio.Event()

    async def sampler() -> None:
        nonlocal vram_total_mb
        while not stop.is_set():
            util, used_mb, total_mb = gpu_metrics(gpu_index)
            util_samples.append(util)
            vram_samples.append(used_mb)
            if total_mb:
                vram_total_mb = total_mb
            await asyncio.sleep(0.5)

    def run_blocking_sync() -> dict[str, Any]:
        import torch
        import whisper

        torch.cuda.set_device(gpu_index)
        torch.cuda.reset_peak_memory_stats(gpu_index)
        model = whisper.load_model(model_name, device=f"cuda:{gpu_index}", download_root=settings.MODEL_CACHE_DIR)
        try:
            all_segments: list[dict[str, Any]] = []
            all_text_parts: list[str] = []

            # Map language to code if possible (whisper supports names/codes; we keep user's value)
            lang = language

            for chunk in chunks:
                result = model.transcribe(
                    str(chunk.path),
                    language=lang,
                    temperature=settings.WHISPER_TEMPERATURE,
                    beam_size=settings.WHISPER_BEAM_SIZE,
                    condition_on_previous_text=settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
                    logprob_threshold=settings.WHISPER_LOGPROB_THRESHOLD,
                    no_speech_threshold=settings.WHISPER_NO_SPEECH_THRESHOLD,
                    compression_ratio_threshold=settings.WHISPER_COMPRESSION_RATIO_THRESHOLD,
                )

                text = str(result.get("text") or "").strip()
                text_pp = postprocess_text(text).strip()
                if text_pp:
                    all_text_parts.append(text_pp)

                segs = result.get("segments") or []
                if isinstance(segs, list):
                    for s in segs:
                        if not isinstance(s, dict):
                            continue
                        start = s.get("start")
                        end = s.get("end")
                        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                            s2 = dict(s)
                            s2["start"] = float(start) + float(chunk.start)
                            s2["end"] = float(end) + float(chunk.start)
                            t = s2.get("text")
                            if isinstance(t, str):
                                s2["text"] = postprocess_text(t)
                            all_segments.append(s2)
                else:
                    # Fallback: chunk-level single segment
                    if text_pp:
                        all_segments.append({"start": float(chunk.start), "end": float(chunk.end), "text": text_pp})
        finally:
            del model
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        text_full = " ".join(all_text_parts).strip()

        token_count = 0
        if text_full:
            lang_code = language
            try:
                to_code = getattr(whisper.tokenizer, "TO_LANGUAGE_CODE", None)
                if isinstance(to_code, dict):
                    lang_code = to_code.get(str(language).lower(), language)
            except Exception:
                pass
            try:
                tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language=lang_code, task="transcribe")
                token_count = len(tokenizer.encode(text_full))
            except Exception:
                token_count = 0

        peak_alloc_mb = float(torch.cuda.max_memory_allocated(gpu_index)) / (1024 * 1024)
        return {
            "text": text_full,
            "segments": all_segments,
            "token_count": token_count,
            "vram_peak_allocated_mb": peak_alloc_mb,
        }

    sampler_task = asyncio.create_task(sampler())
    try:
        res = await asyncio.to_thread(run_blocking_sync)
    finally:
        stop.set()
        try:
            await sampler_task
        except Exception:
            pass

    util, used_mb, total_mb = gpu_metrics(gpu_index)
    if total_mb and not vram_total_mb:
        vram_total_mb = total_mb

    util_avg = sum(util_samples) / len(util_samples) if util_samples else util
    util_max = max(util_samples) if util_samples else util
    vram_used_avg = sum(vram_samples) / len(vram_samples) if vram_samples else used_mb
    vram_used_max = max(vram_samples) if vram_samples else used_mb
    vram_used_pct_max = (vram_used_max / vram_total_mb * 100.0) if vram_total_mb else 0.0
    vram_used_pct = (used_mb / vram_total_mb * 100.0) if vram_total_mb else 0.0

    res["gpu"] = {
        "index": gpu_index,
        "util_avg_percent": util_avg,
        "util_max_percent": util_max,
        "vram_total_mb": vram_total_mb,
        "vram_used_avg_mb": vram_used_avg,
        "vram_used_max_mb": vram_used_max,
        "vram_used_percent": vram_used_pct,
        "vram_used_percent_max": vram_used_pct_max,
    }
    return res
