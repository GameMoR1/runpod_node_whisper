from __future__ import annotations

import re


_CYR_RE = re.compile(r"[а-яА-Я]")


def postprocess_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    out: list[str] = []
    for ln in lines:
        if len(ln) < 4:
            continue
        if not _CYR_RE.search(ln):
            continue
        if _has_triplet_repeat(ln):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _has_triplet_repeat(s: str) -> bool:
    last = ""
    run = 0
    for ch in s:
        if ch == last:
            run += 1
            if run >= 3:
                return True
        else:
            last = ch
            run = 1
    return False

