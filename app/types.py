from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


JobStatus = Literal["queued", "running", "completed", "failed"]
HealthStatus = Literal["starting", "ready", "error"]


@dataclass
class ModelState:
    id_model: int
    model_name: str
    enabled: bool
    status: Literal["queued_for_download", "downloading", "downloaded", "error"]
    progress: float
    error: Optional[str]


@dataclass
class GpuState:
    index: int
    name: str
    util_percent: float
    vram_used_mb: float
    vram_total_mb: float
    vram_used_percent: float
    status: Literal["idle", "running"]
    current_job_id: Optional[str]
    current_model: Optional[str]


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    model: str
    language: str
    callback_url: str

    created_at_ms: int
    started_at_ms: Optional[int]
    finished_at_ms: Optional[int]

    result: Optional[dict[str, Any]]
    error: Optional[str]

    callback_delivered_at_ms: Optional[int]
    callback_error: Optional[str]
    file_dir: str

