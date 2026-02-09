from __future__ import annotations

import asyncio
import os
from pathlib import Path

from app.config import settings


async def preprocess_to_wav(input_path: str, output_path: str) -> None:
    in_p = str(Path(input_path))
    out_p = str(Path(output_path))
    cmd = [
        settings.FFMPEG_PATH,
        "-y",
        "-i",
        in_p,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "silenceremove=start_periods=1:start_silence=0.5:start_threshold=-40dB",
        out_p,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg preprocessing failed")

