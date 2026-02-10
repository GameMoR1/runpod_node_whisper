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

    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASS: str

    DATA_DIR: str = "data"
    UPLOAD_DIR: str = "data/uploads"
    MODEL_CACHE_DIR: str = "data/whisper_models"

    DASHBOARD_REFRESH_MS: int = 2000

    MODEL_DOWNLOAD_ATTEMPTS: int = 3
    MODEL_DOWNLOAD_TIMEOUT_S: int = 30
    MODEL_PREPARE_RETRY_S: int = 60

    WHISPER_DEFAULT_LANGUAGE: str = "Russian"
    WHISPER_TEMPERATURE: float = 0.0
    WHISPER_LOGPROB_THRESHOLD: float = -1.0

    FFMPEG_PATH: str = "ffmpeg"


_apply_env_blob()
settings = Settings()
