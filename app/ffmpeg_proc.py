from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import settings


async def _ensure_rnnoise_model(model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / settings.RNNOISE_MODEL_FILENAME
    if model_path.exists():
        return model_path

    import httpx

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        r = await client.get(settings.RNNOISE_MODEL_URL)
        r.raise_for_status()
        model_path.write_bytes(r.content)
    return model_path


async def preprocess_to_wav(input_path: str, output_path: str) -> dict[str, object]:
    """
    Best pipeline:
    - Convert to wav 16kHz mono
    - Optional RNNoise (std) via FFmpeg arnndn
    """
    in_p = str(Path(input_path))
    out_p = str(Path(output_path))

    filters: list[str] = []
    pipeline_notes: list[str] = []

    if settings.RNNOISE_ENABLED:
        model_path = await _ensure_rnnoise_model(Path(settings.DATA_DIR) / "rnnoise")
        filters.append(f"arnndn=m={model_path.as_posix()}")
        pipeline_notes.append(f"rnnoise=on ({model_path.name})")
    else:
        pipeline_notes.append("rnnoise=off")

    audio_filter = ",".join(filters) if filters else None

    cmd = [
        settings.FFMPEG_PATH,
        "-y",
        "-loglevel",
        "error",
        "-i",
        in_p,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
    ]
    if audio_filter:
        cmd.extend(["-af", audio_filter])
    cmd.append(out_p)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg preprocessing failed")

    return {
        "ffmpeg": {"path": settings.FFMPEG_PATH, "args": cmd},
        "wav": {"sr_hz": 16000, "channels": 1, "codec": "pcm_s16le"},
        "rnnoise": {
            "enabled": bool(settings.RNNOISE_ENABLED),
            "model_url": settings.RNNOISE_MODEL_URL,
            "model_filename": settings.RNNOISE_MODEL_FILENAME,
        },
        "notes": pipeline_notes,
    }

