"""Split text into overlapping chunks of roughly 300 to 500 tokens, plus a file summary."""

from __future__ import annotations

import re

from ..tokens import estimate_tokens

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\n+")


def _units(text: str) -> list[str]:
    """Sentences and lines, the smallest pieces a chunk is built from."""
    return [u.strip() for u in _SENTENCE_END.split(text) if u and u.strip()]


def _split_long(unit: str, max_tokens: int) -> list[str]:
    """Hard-split a unit that alone exceeds the maximum, on word boundaries."""
    words = unit.split()
    pieces: list[str] = []
    current: list[str] = []
    for w in words:
        trial = " ".join(current + [w])
        if current and estimate_tokens(trial) > max_tokens:
            pieces.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    text: str,
    target_tokens: int = 400,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """Greedy sentence packing with overlap.

    A chunk is closed once it reaches ``target_tokens``. The next chunk starts with the
    last few units of the previous one, up to ``overlap_tokens``, so context carries over.
    """
    if target_tokens > max_tokens:
        raise ValueError("target_tokens cannot exceed max_tokens")
    units: list[str] = []
    for u in _units(text):
        if estimate_tokens(u) > max_tokens:
            units.extend(_split_long(u, max_tokens))
        else:
            units.append(u)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    fresh = 0  # units added since the last flush (overlap carried over does not count)

    def flush() -> None:
        nonlocal current, current_tokens, fresh
        chunks.append(" ".join(current))
        current, current_tokens = _overlap_tail(current, overlap_tokens)
        fresh = 0

    for u in units:
        t = estimate_tokens(u)
        if current and current_tokens + t > max_tokens:
            if fresh > 0:
                flush()
            if current_tokens + t > max_tokens:  # the carried overlap alone would break the cap
                current, current_tokens = [], 0
        current.append(u)
        current_tokens += t
        fresh += 1
        if current_tokens >= target_tokens:
            flush()
    if fresh > 0:
        chunks.append(" ".join(current))
    return chunks


def _overlap_tail(units: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    tail: list[str] = []
    total = 0
    for u in reversed(units):
        t = estimate_tokens(u)
        if total + t > overlap_tokens:
            break
        tail.insert(0, u)
        total += t
    return tail, total


def summarize(text: str, max_chars: int = 160) -> str:
    """One-line extractive summary: the first meaningful line, trimmed.

    A placeholder for a model-written summary. It is cheap, local and predictable.
    """
    for line in text.splitlines():
        line = " ".join(line.split())
        if len(line) >= 8:
            return line if len(line) <= max_chars else line[: max_chars - 1].rstrip() + "…"
    flat = " ".join(text.split())
    return flat[:max_chars]
