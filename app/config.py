from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASS: str

    DATA_DIR: str = "data"
    UPLOAD_DIR: str = "data/uploads"
    MODEL_CACHE_DIR: str = "data/whisper_models"

    DASHBOARD_REFRESH_MS: int = 2000

    WHISPER_DEFAULT_LANGUAGE: str = "Russian"
    WHISPER_TEMPERATURE: float = 0.0
    WHISPER_LOGPROB_THRESHOLD: float = -1.0

    FFMPEG_PATH: str = "ffmpeg"


settings = Settings()
