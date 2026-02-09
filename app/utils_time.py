from __future__ import annotations

import time


def now_ms() -> int:
    return int(time.time() * 1000)


def ms_to_s(ms: int) -> float:
    return ms / 1000.0

