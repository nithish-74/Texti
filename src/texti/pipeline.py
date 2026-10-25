"""Process one local file into encrypted, tiered, searchable chunks.

Order matters for privacy: classify and mask each page BEFORE chunking, embedding or
storing, so sensitive values never reach the embedder or the database.

Logging records only opaque ids, counts and timings. It never records file names, paths
or text.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .extract import extract_file
from .index.chunker import chunk_text, summarize
from .index.embedder import Embedder
from .index.store import ChunkRecord, Store
from .privacy.classifier import PathRule, classify_chunk, document_base_tier
from .privacy.redact import redact
from .retrieve.budget import estimate_raw_tokens
from .tokens import estimate_tokens

log = logging.getLogger("texti")

_EMBED_BATCH = 64


@dataclass
class ProcessResult:
    file_id: str
    status: str  # 'done', 'unchanged', 'empty'
    max_tier: int = 0
    chunks: int = 0
    chunks_by_tier: dict[int, int] = field(default_factory=dict)
    seconds: float = 0.0
    reasons: list[str] = field(default_factory=list)


def process_file(
    store: Store,
    embedder: Embedder,
    cfg: Config,
    *,
    path: Path,
    file_id: str,
    rel_path: str,
    content_hash: str,
    language: str = "eng",
    rules: list[PathRule] | None = None,
    force: bool = False,
) -> ProcessResult:
    t0 = time.perf_counter()
    if not force and store.file_hash(file_id) == content_hash:
        return ProcessResult(file_id, "unchanged")

    extraction = extract_file(path, language)
    rules = cfg.rules() if rules is None else rules
    full_text = "\n".join(s.text for s in extraction.segments)
    base_tier, reasons = document_base_tier(full_text, rel_path, rules, cfg.default_tier)
    raw_tokens = estimate_raw_tokens(Path(rel_path).suffix, extraction.pages, cfg.image_tokens)

    # Replace any previous version of this file, in every tier.
    store.delete_file_chunks(file_id)

    pending: list[tuple[int, str, int, float, str]] = []  # (tier, text, offset, confidence, level)
    summary_done = False
    max_tier = base_tier
    for seg in extraction.segments:
        decision = classify_chunk(seg.text, base_tier)
        tier = decision.tier
        max_tier = max(max_tier, tier)
        masked = redact(seg.text, tier, decision.findings)
        if not summary_done and masked.strip():
            pending.append((tier, summarize(masked), seg.offset, seg.confidence, "summary"))
            summary_done = True
        for piece in chunk_text(masked, cfg.target_tokens, cfg.max_tokens, cfg.overlap_tokens):
            pending.append((tier, piece, seg.offset, seg.confidence, "full"))

    by_tier: dict[int, list[ChunkRecord]] = {}
    for i in range(0, len(pending), _EMBED_BATCH):
        batch = pending[i : i + _EMBED_BATCH]
        vectors = embedder.embed([b[1] for b in batch])
        for (tier, text, offset, conf, level), vec in zip(batch, vectors):
            by_tier.setdefault(tier, []).append(
                ChunkRecord(
                    file_id=file_id,
                    level=level,
                    text=text,
                    source_path=rel_path,
                    embedding=vec,
                    token_count=estimate_tokens(text),
                    source_offset=offset,
                    language=language,
                    confidence=conf,
                )
            )
    for tier, records in by_tier.items():
        store.add_chunks(tier, records)

    seconds = time.perf_counter() - t0
    status = "done" if pending else "empty"
    store.upsert_file(file_id, content_hash, max_tier, raw_tokens, seconds, "done")
    counts = {t: len(r) for t, r in by_tier.items()}
    log.info("processed file_id=%s status=%s max_tier=%d chunks=%d seconds=%.2f",
             file_id, status, max_tier, len(pending), seconds)
    return ProcessResult(file_id, status, max_tier, len(pending), counts, seconds, reasons)
