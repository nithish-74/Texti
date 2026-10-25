"""Token budgeter: pick the cheapest useful text that fits a token budget.

Order of preference: a one-line summary for each of the best files first, then full
chunks in score order. The saved-token figure compares what was returned against an
estimate of what the raw media would have cost, and it is an estimate wherever it is shown.
"""

from __future__ import annotations

from collections import defaultdict

from ..index.store import Candidate

from ..extract.audio import MEDIA_AUDIO_EXTENSIONS

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


def estimate_raw_tokens(extension: str, pages: int, tokens_per_image: int) -> int:
    """Rough cost of sending the original as image(s) or audio: one image per page or 1500 tokens/min of audio.

    The default per-image figure (765) is a published GPT-4o estimate for a 1280x720
    image. Real cost varies by model and resolution, so it lives in the config.
    """
    ext = extension.lower()
    if ext in IMAGE_EXTENSIONS:
        return tokens_per_image
    if ext == ".pdf":
        return max(1, pages) * tokens_per_image
    if ext in MEDIA_AUDIO_EXTENSIONS:
        return max(1, pages) * 1500
    return 0



def select_within_budget(
    scored: list[tuple[Candidate, float]],
    budget: int,
    min_score: float = 0.1,
    max_files: int = 5,
) -> list[tuple[Candidate, float]]:
    """Choose candidates whose token counts fit ``budget``; return them grouped by file."""
    by_file: dict[tuple[int, str], list[tuple[Candidate, float]]] = defaultdict(list)
    for c, s in scored:
        by_file[(c.tier, c.file_id)].append((c, s))

    ranked = sorted(
        ((max(s for _, s in items), key, items) for key, items in by_file.items()),
        key=lambda t: t[0],
        reverse=True,
    )
    ranked = [r for r in ranked if r[0] >= min_score][:max_files]

    chosen: dict[tuple[int, str], list[tuple[Candidate, float]]] = {key: [] for _, key, _ in ranked}
    used = 0

    # Pass 1: one summary per relevant file.
    for _, key, items in ranked:
        summaries = [it for it in items if it[0].level == "summary"]
        if summaries:
            best = max(summaries, key=lambda it: it[1])
            if used + best[0].token_count <= budget:
                chosen[key].append(best)
                used += best[0].token_count

    # Pass 2: full chunks, best score first, skipping any that no longer fit.
    fulls = [
        (key, it)
        for _, key, items in ranked
        for it in items
        if it[0].level == "full" and it[1] >= min_score
    ]
    for key, it in sorted(fulls, key=lambda t: t[1][1], reverse=True):
        if used + it[0].token_count <= budget:
            chosen[key].append(it)
            used += it[0].token_count

    ordered: list[tuple[Candidate, float]] = []
    for _, key, _ in ranked:
        items = chosen[key]
        # A summary only helps when the file's full text is not included; otherwise it
        # repeats words the reader already has and wastes tokens.
        if any(c.level == "full" for c, _ in items):
            items = [it for it in items if it[0].level != "summary"]
        # summary first, then full chunks in document order
        items.sort(key=lambda it: (it[0].level != "summary", it[0].source_offset, it[0].chunk_id))
        ordered.extend(items)
    return ordered
