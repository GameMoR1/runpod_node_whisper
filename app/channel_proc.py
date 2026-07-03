from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.config import settings
from app.ffmpeg_proc import preprocess_to_wav
from app.vad_chunking import AudioChunk, build_hybrid_chunks, get_speech_segments


def _ffprobe_path() -> str:
    ffmpeg = settings.FFMPEG_PATH
    if ffmpeg.lower().endswith("ffmpeg.exe"):
        return ffmpeg[:-10] + "ffprobe.exe"
    if ffmpeg.lower().endswith("ffmpeg"):
        return ffmpeg[:-6] + "ffprobe"
    return "ffprobe"


async def probe_audio_channels(input_path: str) -> int:
    cmd = [
        _ffprobe_path(),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _stderr = await proc.communicate()
        if proc.returncode != 0:
            return 1
        value = (stdout or b"").decode("utf-8", errors="replace").strip()
        channels = int(value)
        return max(1, channels)
    except Exception:
        return 1


async def preprocess_channel_to_wav(input_path: str, output_path: str, channel_index: int) -> dict[str, object]:
    from app.ffmpeg_proc import _ensure_rnnoise_model

    in_p = str(Path(input_path))
    out_p = str(Path(output_path))
    filters: list[str] = [f"pan=mono|c0=c{channel_index}"]
    pipeline_notes = [f"channel={channel_index}"]

    if settings.RNNOISE_ENABLED:
        model_path = await _ensure_rnnoise_model(Path(settings.DATA_DIR) / "rnnoise")
        filters.append(f"arnndn=m={model_path.as_posix()}")
        pipeline_notes.append(f"rnnoise=on ({model_path.name})")

    cmd = [
        settings.FFMPEG_PATH,
        "-y",
        "-loglevel",
        "error",
        "-i",
        in_p,
        "-af",
        ",".join(filters),
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        out_p,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        err_text = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg channel preprocess failed: {err_text or proc.returncode}")

    return {
        "ffmpeg": {"path": settings.FFMPEG_PATH, "args": cmd},
        "wav": {"sr_hz": 16000, "channels": 1, "codec": "pcm_s16le", "source_channel": channel_index},
        "notes": pipeline_notes,
    }


async def prepare_channel_wavs(
    input_path: str,
    job_dir: Path,
    *,
    split_by_channels: bool,
) -> tuple[list[tuple[int, Path]], dict[str, Any]]:
    source_channels = await probe_audio_channels(input_path)
    if not split_by_channels or source_channels <= 1:
        wav_path = job_dir / "audio.wav"
        info = await preprocess_to_wav(input_path, str(wav_path))
        return [(0, wav_path)], {
            "mode": "mono",
            "source_channels": source_channels,
            "split_by_channels": bool(split_by_channels),
            "preprocess": info,
        }

    sources: list[tuple[int, Path]] = []
    channel_infos: list[dict[str, Any]] = []
    for ch in range(source_channels):
        wav_path = job_dir / f"audio_ch{ch}.wav"
        info = await preprocess_channel_to_wav(input_path, str(wav_path), ch)
        sources.append((ch, wav_path))
        channel_infos.append(info)

    return sources, {
        "mode": "split_channels",
        "source_channels": source_channels,
        "split_by_channels": True,
        "channels": channel_infos,
    }


def merge_channel_results(channel_results: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    speakers: list[dict[str, Any]] = []
    all_segments: list[dict[str, Any]] = []
    text_parts: list[str] = []
    token_count = 0
    gpu: dict[str, Any] | None = None
    vram_peak = 0.0

    for ch_idx, res in channel_results:
        speaker = f"speaker_{ch_idx}"
        speakers.append({"id": speaker, "channel": ch_idx})
        token_count += int(res.get("token_count") or 0)
        vram_peak = max(vram_peak, float(res.get("vram_peak_allocated_mb") or 0.0))
        if gpu is None and isinstance(res.get("gpu"), dict):
            gpu = dict(res["gpu"])

        for seg in res.get("segments") or []:
            if not isinstance(seg, dict):
                continue
            seg2 = dict(seg)
            seg2["speaker"] = speaker
            seg2["channel"] = ch_idx
            text = str(seg2.get("text") or "").strip()
            if text:
                all_segments.append(seg2)

        ch_text = str(res.get("text") or "").strip()
        if ch_text:
            text_parts.append(ch_text)

    all_segments.sort(key=lambda s: (float(s.get("start") or 0.0), int(s.get("channel") or 0)))
    if all_segments:
        text = " ".join(str(s.get("text") or "").strip() for s in all_segments if s.get("text")).strip()
    else:
        text = " ".join(text_parts).strip()

    merged: dict[str, Any] = {
        "text": text,
        "segments": all_segments,
        "speakers": speakers,
        "token_count": token_count,
        "vram_peak_allocated_mb": vram_peak,
    }
    if gpu is not None:
        merged["gpu"] = gpu
    return merged


async def transcribe_channel_wav(
    *,
    gpu_index: int,
    wav_path: Path,
    job_dir: Path,
    model_name: str,
    language: str,
    transcribe_chunks_fn: Callable[..., Awaitable[dict[str, Any]]],
    channel_index: int,
    tag_speaker: bool,
    min_chunk_duration: float = 0.0,
) -> dict[str, Any]:
    segments = get_speech_segments(wav_path)
    chunks: list[AudioChunk] = build_hybrid_chunks(audio_path=wav_path, segments=segments, job_dir=job_dir)
    if min_chunk_duration > 0:
        chunks = [c for c in chunks if c.duration > min_chunk_duration]
    if not chunks:
        return {"text": "", "segments": [], "token_count": 0}

    result = await transcribe_chunks_fn(
        gpu_index=gpu_index,
        chunks=chunks,
        model_name=model_name,
        language=language,
    )
    if tag_speaker:
        speaker = f"speaker_{channel_index}"
        for seg in result.get("segments") or []:
            if isinstance(seg, dict):
                seg["speaker"] = speaker
                seg["channel"] = channel_index
        if result.get("text"):
            result["speakers"] = [{"id": speaker, "channel": channel_index}]
    return result
