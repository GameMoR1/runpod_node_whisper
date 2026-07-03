from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _is_running_in_pod() -> bool:
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    if os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_SERVERLESS"):
        return True
    return False


def _dotenv_file() -> str | None:
    if os.environ.get("DISABLE_DOTENV") == "1":
        return None
    if _is_running_in_pod():
        return None
    return ".env" if Path(".env").exists() else None


def _secrets_dirs() -> tuple[str, ...] | None:
    raw = os.environ.get("SECRETS_DIR")
    if raw:
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        return tuple(parts) if parts else None

    candidates = [
        "/run/secrets",
        "/var/run/secrets",
        "/var/run/secrets/kubernetes.io",
        "/etc/secrets",
    ]
    existing = [p for p in candidates if Path(p).exists()]
    return tuple(existing) if existing else None


def _apply_env_blob() -> None:
    raw = os.environ.get("ENVIRONMENT_VARIABLE")
    if not raw:
        raw = os.environ.get("RUNPOD_SECRET_ENV_TRANSCRIBERS")
    if not raw:
        return

    for line in str(raw).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_dotenv_file(),
        env_file_encoding="utf-8",
        secrets_dir=_secrets_dirs(),
        extra="ignore",
    )

    # DB settings are required in production, but optional for local import/tests.
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "postgres"
    DB_USER: str = "postgres"
    DB_PASS: str = "postgres"

    DATA_DIR: str = "data"
    UPLOAD_DIR: str = "data/uploads"
    MODEL_CACHE_DIR: str = "data/whisper_models"

    DASHBOARD_REFRESH_MS: int = 2000

    MODEL_DOWNLOAD_ATTEMPTS: int = 3
    MODEL_DOWNLOAD_TIMEOUT_S: int = 30
    MODEL_PREPARE_RETRY_S: int = 60

    CALLBACK_TIMEOUT_S: float = 120.0
    CALLBACK_ATTEMPTS: int = 4

    WHISPER_DEFAULT_LANGUAGE: str = "Russian"
    # Best params from internal experiments report:
    # beam_size=3, temperature=0, condition_on_previous_text=False
    WHISPER_TEMPERATURE: float = 0.0
    WHISPER_BEAM_SIZE: int = 3
    WHISPER_CONDITION_ON_PREVIOUS_TEXT: bool = False
    WHISPER_LOGPROB_THRESHOLD: float | None = None
    WHISPER_NO_SPEECH_THRESHOLD: float | None = None
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float | None = None

    # Audio pipeline (best):
    # 1) FFmpeg wav 16kHz mono
    # 2) RNNoise (std) via FFmpeg arnndn
    # 3) VAD (pyannote preferred)
    # 4) Hybrid chunking
    RNNOISE_ENABLED: bool = True
    RNNOISE_MODEL_URL: str = "https://raw.githubusercontent.com/richardpl/arnndn-models/master/std.rnnn"
    RNNOISE_MODEL_FILENAME: str = "rnnoise_std.rnnn"

    VAD_METHOD: str = "pyannote"  # pyannote / silero / off
    PYANNOTE_MODEL: str = "pyannote/voice-activity-detection"
    PYANNOTE_AUTH_TOKEN: str = ""

    SILERO_VAD_THRESHOLD: float = 0.5
    SILERO_MIN_SPEECH_DURATION_MS: int = 250
    SILERO_MIN_SILENCE_DURATION_MS: int = 300

    VAD_PADDING_SECONDS: float = 0.25
    CHUNK_MIN_SECONDS: float = 3.0
    CHUNK_MAX_SECONDS: float = 30.0
    CHUNK_MERGE_GAP_SECONDS: float = 0.8
    CHUNK_SPLIT_OVERLAP_SECONDS: float = 2.0

    FFMPEG_PATH: str = "ffmpeg"


_apply_env_blob()
settings = Settings()
