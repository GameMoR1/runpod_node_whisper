from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings


@dataclass(frozen=True)
class AudioSegment:
    start: float
    end: float
    source: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def get_audio_duration_s(audio_path: Path) -> float:
    import wave

    with wave.open(str(audio_path), "rb") as wf:
        fr = wf.getframerate()
        nframes = wf.getnframes()
        if fr <= 0:
            return 0.0
        return float(nframes) / float(fr)


_silero_model: Any | None = None
_pyannote_pipeline: Any | None = None


def get_speech_segments(audio_path: Path) -> list[AudioSegment]:
    method = str(settings.VAD_METHOD or "off").lower().strip()
    if method in {"off", "none", "no"}:
        dur = get_audio_duration_s(audio_path)
        return [AudioSegment(start=0.0, end=dur, source="no_vad")]
    if method == "silero":
        return _get_silero_segments(audio_path)
    if method == "pyannote":
        try:
            return _get_pyannote_segments(audio_path)
        except Exception:
            # If pyannote isn't available/token missing, fall back to Silero if possible.
            return _get_silero_segments(audio_path)
    raise ValueError(f"Unknown VAD_METHOD: {settings.VAD_METHOD}")


def _get_silero_segments(audio_path: Path) -> list[AudioSegment]:
    global _silero_model
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

    if _silero_model is None:
        _silero_model = load_silero_vad()

    wav = read_audio(str(audio_path), sampling_rate=16000)
    timestamps = get_speech_timestamps(
        wav,
        _silero_model,
        sampling_rate=16000,
        threshold=settings.SILERO_VAD_THRESHOLD,
        min_speech_duration_ms=settings.SILERO_MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=settings.SILERO_MIN_SILENCE_DURATION_MS,
    )
    segs: list[AudioSegment] = []
    for item in timestamps:
        start = float(item["start"]) / 16000.0
        end = float(item["end"]) / 16000.0
        if end > start:
            segs.append(AudioSegment(start=start, end=end, source="silero"))
    if not segs:
        dur = get_audio_duration_s(audio_path)
        return [AudioSegment(start=0.0, end=dur, source="silero_empty_fallback")]
    return segs


def _get_pyannote_segments(audio_path: Path) -> list[AudioSegment]:
    global _pyannote_pipeline

    token = (settings.PYANNOTE_AUTH_TOKEN or "").strip()
    if not token:
        raise RuntimeError("PYANNOTE_AUTH_TOKEN is empty")

    if _pyannote_pipeline is None:
        from pyannote.audio import Pipeline

        _pyannote_pipeline = Pipeline.from_pretrained(settings.PYANNOTE_MODEL, use_auth_token=token)

    result = _pyannote_pipeline(str(audio_path))
    segs: list[AudioSegment] = []
    for speech_turn in result.get_timeline().support():
        start = float(speech_turn.start)
        end = float(speech_turn.end)
        if end > start:
            segs.append(AudioSegment(start=start, end=end, source="pyannote"))
    if not segs:
        dur = get_audio_duration_s(audio_path)
        return [AudioSegment(start=0.0, end=dur, source="pyannote_empty_fallback")]
    return segs


def build_hybrid_chunks(*, audio_path: Path, segments: list[AudioSegment], job_dir: Path) -> list[AudioChunk]:
    duration = get_audio_duration_s(audio_path)

    # Merge short/close segments
    sorted_segments = sorted(segments, key=lambda s: s.start)
    merged: list[AudioSegment] = []
    if sorted_segments:
        cur_start = sorted_segments[0].start
        cur_end = sorted_segments[0].end
        for seg in sorted_segments[1:]:
            gap = seg.start - cur_end
            cur_dur = cur_end - cur_start
            should_merge = gap <= settings.CHUNK_MERGE_GAP_SECONDS or cur_dur < settings.CHUNK_MIN_SECONDS
            if should_merge and (seg.end - cur_start) <= settings.CHUNK_MAX_SECONDS:
                cur_end = max(cur_end, seg.end)
            else:
                merged.append(AudioSegment(start=cur_start, end=cur_end, source="vad_hybrid"))
                cur_start = seg.start
                cur_end = seg.end
        merged.append(AudioSegment(start=cur_start, end=cur_end, source="vad_hybrid"))

    # Split long segments
    final: list[AudioSegment] = []
    for seg in merged:
        if seg.duration <= settings.CHUNK_MAX_SECONDS:
            final.append(seg)
            continue
        start = seg.start
        overlap = min(2.0, float(settings.CHUNK_SPLIT_OVERLAP_SECONDS))
        while start < seg.end:
            end = min(seg.end, start + settings.CHUNK_MAX_SECONDS)
            final.append(AudioSegment(start=start, end=end, source="vad_hybrid_split"))
            if end >= seg.end:
                break
            start = max(start + 0.01, end - overlap)

    out_dir = job_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[AudioChunk] = []
    for idx, seg in enumerate(final):
        start = max(0.0, seg.start - settings.VAD_PADDING_SECONDS)
        end = min(duration, seg.end + settings.VAD_PADDING_SECONDS)
        if end <= start + 0.01:
            continue
        out_path = out_dir / f"chunk_{idx:04d}_{start:.2f}_{end:.2f}.wav"
        if not out_path.exists():
            _extract_chunk_ffmpeg(audio_path=audio_path, out_path=out_path, start=start, end=end)
        chunks.append(AudioChunk(path=out_path, start=start, end=end))

    if not chunks:
        chunks = [AudioChunk(path=audio_path, start=0.0, end=duration)]
    return chunks


def _extract_chunk_ffmpeg(*, audio_path: Path, out_path: Path, start: float, end: float) -> None:
    import subprocess

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.01, end - start)
    cmd = [
        settings.FFMPEG_PATH,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def pipeline_info() -> dict[str, Any]:
    return {
        "preprocess": {
            "wav": {"sr_hz": 16000, "channels": 1},
            "rnnoise": {
                "enabled": bool(settings.RNNOISE_ENABLED),
                "model_url": settings.RNNOISE_MODEL_URL,
                "model_filename": settings.RNNOISE_MODEL_FILENAME,
            },
        },
        "vad": {
            "method": settings.VAD_METHOD,
            "pyannote_model": settings.PYANNOTE_MODEL,
            "pyannote_token_set": bool((settings.PYANNOTE_AUTH_TOKEN or "").strip()),
            "silero": {
                "threshold": settings.SILERO_VAD_THRESHOLD,
                "min_speech_ms": settings.SILERO_MIN_SPEECH_DURATION_MS,
                "min_silence_ms": settings.SILERO_MIN_SILENCE_DURATION_MS,
            },
        },
        "chunking": {
            "mode": "vad_hybrid",
            "padding_s": settings.VAD_PADDING_SECONDS,
            "min_s": settings.CHUNK_MIN_SECONDS,
            "max_s": settings.CHUNK_MAX_SECONDS,
            "merge_gap_s": settings.CHUNK_MERGE_GAP_SECONDS,
            "split_overlap_s": settings.CHUNK_SPLIT_OVERLAP_SECONDS,
        },
        "asr": {
            "model_family": "whisper",
            "beam_size": settings.WHISPER_BEAM_SIZE,
            "temperature": settings.WHISPER_TEMPERATURE,
            "condition_on_previous_text": settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        },
    }

