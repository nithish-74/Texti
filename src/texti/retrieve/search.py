"""Tier-aware search: the only code allowed to read chunks from the store.

Tier rules enforced here:
- Tier 1 is returned directly.
- Tier 2 and Tier 3 are returned only if the caller passes an ``approver`` callback that
  shows the user exactly what would be sent and returns the subset they accept.
  Without an approver those tiers are not even loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..index.embedder import Embedder
from ..index.store import TIERS, Candidate, Store
from .bm25 import compute_bm25_scores
from .budget import select_within_budget

Approver = Callable[[list["Hit"]], list["Hit"]]



@dataclass
class Hit:
    tier: int
    chunk_id: int
    file_id: str
    level: str
    score: float
    text: str
    source_path: str
    source_offset: int
    token_count: int
    confidence: float
    needs_original: bool

    def __repr__(self) -> str:  # never print text in logs or tracebacks
        return f"Hit(tier={self.tier}, file_id={self.file_id!r}, level={self.level!r}, score={self.score:.3f})"


@dataclass
class SearchResult:
    hits: list[Hit]
    tokens_returned: int
    tokens_raw_estimate: int
    tokens_saved_estimate: int
    notes: list[str] = field(default_factory=list)


def search(
    store: Store,
    embedder: Embedder,
    query: str,
    budget: int = 800,
    tiers: tuple[int, ...] = (1,),
    min_confidence: float = 0.6,
    min_score: float = 0.1,
    max_files: int = 5,
    approver: Approver | None = None,
    record_usage: bool = True,
    hybrid_weight: float = 0.3,
) -> SearchResult:
    notes: list[str] = []
    wanted = sorted(set(tiers))
    for t in wanted:
        if t not in TIERS:
            raise ValueError(f"unknown tier {t!r}")

    loadable: list[int] = []
    for t in wanted:
        if t >= 2 and approver is None:
            notes.append(f"Tier {t} was requested but skipped: it needs the user's approval before it is returned.")
        else:
            loadable.append(t)
    if 3 in loadable:
        notes.append("Warning: this search includes Tier 3 (private) data. Approved text will leave the device "
                     "if it is sent to an outside AI.")

    q = embedder.embed([query])[0]
    candidates: list[Candidate] = []
    for t in loadable:
        candidates.extend(store.load_candidates(t, with_text=True))
    if not candidates:
        return SearchResult([], 0, 0, 0, notes)

    # The same file is often stored twice (a "- Copy" folder, a re-download). Identical text gives an
    # identical embedding, so keep one copy: duplicates waste the budget and inflate "tokens saved".
    seen: set[tuple] = set()
    unique: list[Candidate] = []
    for c in candidates:
        key = (c.tier, c.level, c.token_count, c.embedding.tobytes())
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    if len(unique) < len(candidates):
        notes.append(f"{len(candidates) - len(unique)} duplicate copy(ies) of the same text were hidden.")
    candidates = unique

    matrix = np.vstack([c.embedding for c in candidates])
    dense_scores = matrix @ q

    if hybrid_weight > 0.0:
        docs = [c.text for c in candidates]
        lengths = [c.token_count for c in candidates]
        bm25 = compute_bm25_scores(query, docs, doc_lengths=lengths)
        max_bm25 = max(bm25) if bm25 else 0.0
        if max_bm25 > 0.0:
            norm_bm25 = [s / max_bm25 for s in bm25]
            scores = [
                (1.0 - hybrid_weight) * max(0.0, float(ds)) + hybrid_weight * float(bs)
                for ds, bs in zip(dense_scores, norm_bm25)
            ]
        else:
            scores = [float(s) for s in dense_scores]
    else:
        scores = [float(s) for s in dense_scores]

    scored = list(zip(candidates, scores))
    selected = select_within_budget(scored, budget, min_score=min_score, max_files=max_files)


    hits: list[Hit] = []
    for cand, score in selected:
        text, path = store.read_chunk(cand.tier, cand.chunk_id)
        hits.append(
            Hit(
                tier=cand.tier,
                chunk_id=cand.chunk_id,
                file_id=cand.file_id,
                level=cand.level,
                score=score,
                text=text,
                source_path=path,
                source_offset=cand.source_offset,
                token_count=cand.token_count,
                confidence=cand.confidence,
                needs_original=cand.confidence < min_confidence,
            )
        )

    gated = [h for h in hits if h.tier >= 2]
    if gated:
        assert approver is not None  # guaranteed: gated tiers are only loaded with an approver
        accepted = {(h.tier, h.chunk_id) for h in approver(list(gated))}
        hits = [h for h in hits if h.tier == 1 or (h.tier, h.chunk_id) in accepted]
        if len(accepted) < len(gated):
            notes.append("Some results were not approved and were left out.")

    returned = sum(h.token_count for h in hits)
    raw = sum(store.raw_tokens(fid) for fid in {h.file_id for h in hits})
    saved = raw - returned
    if record_usage and hits:
        store.record_usage(returned, raw, saved)
    return SearchResult(hits, returned, raw, saved, notes)
