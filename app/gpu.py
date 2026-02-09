from __future__ import annotations

from dataclasses import asdict
from typing import Optional

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

try:
    import pynvml

    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


def gpu_count() -> int:
    if _NVML_OK:
        try:
            return int(pynvml.nvmlDeviceGetCount())
        except Exception:
            pass
    if not _HAS_TORCH:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def torch_cuda_available() -> bool:
    if not _HAS_TORCH:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_cuda_device_count() -> int:
    if not _HAS_TORCH:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def gpu_name(index: int) -> str:
    if not _HAS_TORCH:
        return f"GPU {index}"
    try:
        return str(torch.cuda.get_device_name(index))
    except Exception:
        return f"GPU {index}"


def _nvml_init() -> bool:
    if not _HAS_NVML:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


_NVML_OK = _nvml_init()


def gpu_metrics(index: int) -> tuple[float, float, float]:
    util = 0.0
    used_mb = 0.0
    total_mb = 0.0
    if not _NVML_OK:
        if not _HAS_TORCH:
            return (0.0, 0.0, 0.0)
        try:
            torch.cuda.set_device(index)
            total_mb = float(torch.cuda.get_device_properties(index).total_memory) / (1024 * 1024)
            used_mb = float(torch.cuda.memory_allocated(index)) / (1024 * 1024)
        except Exception:
            return (0.0, 0.0, 0.0)
        return (util, used_mb, total_mb)
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = float(u.gpu)
        used_mb = float(m.used) / (1024 * 1024)
        total_mb = float(m.total) / (1024 * 1024)
        return (util, used_mb, total_mb)
    except Exception:
        return (0.0, 0.0, 0.0)
