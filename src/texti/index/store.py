"""Encrypted per-tier storage.

Layout inside the data directory:

    tier1.db, tier2.db, tier3.db   one SQLite file per tier, each with its own AES-GCM key
    meta.db                        file bookkeeping with no names, paths or text

Everything that reveals content (chunk text, source path, embedding) is encrypted before
it is written. Embeddings are encrypted too, since text can be partly reconstructed from
them, so vector search is done in memory after decrypting: a brute-force cosine search,
which is fast enough for tens of thousands of chunks. (sqlite-vec was dropped because it
cannot search encrypted vectors.)

Only retrieve/search.py may call ``load_candidates``; tests enforce that.
"""

from __future__ import annotations

import base64
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..secrets_store import SecretStore

TIERS = (1, 2, 3)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       TEXT NOT NULL,
    level         TEXT NOT NULL CHECK (level IN ('summary', 'full')),
    token_count   INTEGER NOT NULL,
    source_offset INTEGER NOT NULL DEFAULT 0,
    language      TEXT NOT NULL DEFAULT '',
    confidence    REAL NOT NULL DEFAULT 1.0,
    enc_text      BLOB NOT NULL,
    enc_path      BLOB NOT NULL,
    enc_embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
"""

_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_id      TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    max_tier     INTEGER NOT NULL,
    raw_tokens   INTEGER NOT NULL DEFAULT 0,
    seconds      REAL NOT NULL DEFAULT 0,
    status       TEXT NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sync_state (
    folder_id   TEXT PRIMARY KEY,
    delta_link  TEXT,
    last_synced REAL
);
CREATE TABLE IF NOT EXISTS usage (
    query_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    at                   REAL NOT NULL,
    tokens_returned      INTEGER NOT NULL,
    tokens_raw_estimate  INTEGER NOT NULL,
    tokens_saved_estimate INTEGER NOT NULL
);
"""


class CryptoError(Exception):
    """Decryption failed: wrong key or tampered data."""


@dataclass
class ChunkRecord:
    file_id: str
    level: str  # 'summary' or 'full'
    text: str
    source_path: str
    embedding: np.ndarray
    token_count: int
    source_offset: int = 0
    language: str = ""
    confidence: float = 1.0


@dataclass
class Candidate:
    """A chunk's searchable fields for dense vector and hybrid BM25 retrieval."""

    tier: int
    chunk_id: int
    file_id: str
    level: str
    token_count: int
    source_offset: int
    language: str
    confidence: float
    embedding: np.ndarray
    text: str = ""



class Store:
    def __init__(self, data_dir: Path, secrets: SecretStore) -> None:
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._secrets = secrets
        self._aes: dict[int, AESGCM] = {}
        self._conns: dict[int, sqlite3.Connection] = {}
        self._meta = sqlite3.connect(self.dir / "meta.db")
        self._meta.executescript(_META_SCHEMA)
        self._restrict(self.dir / "meta.db")

    # -- keys and encryption ----------------------------------------------------

    def _key(self, tier: int) -> AESGCM:
        if tier not in self._aes:
            name = f"tier{tier}-key"
            raw = self._secrets.get(name)
            if raw is None:
                raw = base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")
                self._secrets.set(name, raw)
            self._aes[tier] = AESGCM(base64.b64decode(raw))
        return self._aes[tier]

    def _encrypt(self, tier: int, kind: str, data: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + self._key(tier).encrypt(nonce, data, f"m2t:v1:t{tier}:{kind}".encode())

    def _decrypt(self, tier: int, kind: str, blob: bytes) -> bytes:
        try:
            return self._key(tier).decrypt(blob[:12], blob[12:], f"m2t:v1:t{tier}:{kind}".encode())
        except InvalidTag as e:
            raise CryptoError("could not decrypt: wrong key or corrupted data") from e

    # -- connections --------------------------------------------------------------

    @staticmethod
    def _restrict(path: Path) -> None:
        try:
            os.chmod(path, 0o600)
        except OSError:  # not supported on every filesystem
            pass

    def _db(self, tier: int) -> sqlite3.Connection:
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}")
        if tier not in self._conns:
            path = self.dir / f"tier{tier}.db"
            conn = sqlite3.connect(path)
            conn.executescript(_SCHEMA)
            self._restrict(path)
            self._conns[tier] = conn
        return self._conns[tier]

    def close(self) -> None:
        for c in self._conns.values():
            c.close()
        self._conns.clear()
        self._meta.close()

    # -- writing ------------------------------------------------------------------

    def add_chunks(self, tier: int, records: list[ChunkRecord]) -> None:
        conn = self._db(tier)
        rows = [
            (
                r.file_id,
                r.level,
                r.token_count,
                r.source_offset,
                r.language,
                r.confidence,
                self._encrypt(tier, "text", r.text.encode("utf-8")),
                self._encrypt(tier, "path", r.source_path.encode("utf-8")),
                self._encrypt(tier, "emb", np.asarray(r.embedding, dtype=np.float32).tobytes()),
            )
            for r in records
        ]
        with conn:
            conn.executemany(
                "INSERT INTO chunks (file_id, level, token_count, source_offset, language, confidence,"
                " enc_text, enc_path, enc_embedding) VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def delete_file_chunks(self, file_id: str) -> None:
        """Remove a file's chunks from every tier (used on re-sync, tier change and deletion)."""
        for tier in TIERS:
            if (self.dir / f"tier{tier}.db").exists() or tier in self._conns:
                with self._db(tier) as conn:
                    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

    # -- reading (search module only) ----------------------------------------------

    def load_candidates(self, tier: int, with_text: bool = True) -> list[Candidate]:
        conn = self._db(tier)
        if with_text:
            rows = conn.execute(
                "SELECT chunk_id, file_id, level, token_count, source_offset, language, confidence,"
                " enc_embedding, enc_text FROM chunks"
            ).fetchall()
            return [
                Candidate(
                    tier,
                    cid,
                    fid,
                    level,
                    tc,
                    off,
                    lang,
                    conf,
                    np.frombuffer(self._decrypt(tier, "emb", emb), dtype=np.float32),
                    self._decrypt(tier, "text", enc_t).decode("utf-8"),
                )
                for cid, fid, level, tc, off, lang, conf, emb, enc_t in rows
            ]
        rows = conn.execute(
            "SELECT chunk_id, file_id, level, token_count, source_offset, language, confidence,"
            " enc_embedding FROM chunks"
        ).fetchall()
        return [
            Candidate(
                tier,
                cid,
                fid,
                level,
                tc,
                off,
                lang,
                conf,
                np.frombuffer(self._decrypt(tier, "emb", emb), dtype=np.float32),
            )
            for cid, fid, level, tc, off, lang, conf, emb in rows
        ]


    def read_chunk(self, tier: int, chunk_id: int) -> tuple[str, str]:
        """Decrypted (text, source_path) of one selected chunk."""
        row = self._db(tier).execute(
            "SELECT enc_text, enc_path FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            raise KeyError(chunk_id)
        return (
            self._decrypt(tier, "text", row[0]).decode("utf-8"),
            self._decrypt(tier, "path", row[1]).decode("utf-8"),
        )

    # -- bookkeeping (no content) ----------------------------------------------------

    def upsert_file(self, file_id: str, content_hash: str, max_tier: int, raw_tokens: int,
                    seconds: float, status: str) -> None:
        with self._meta:
            self._meta.execute(
                "INSERT INTO files (file_id, content_hash, max_tier, raw_tokens, seconds, status, updated_at)"
                " VALUES (?,?,?,?,?,?,?) ON CONFLICT(file_id) DO UPDATE SET content_hash=excluded.content_hash,"
                " max_tier=excluded.max_tier, raw_tokens=excluded.raw_tokens, seconds=excluded.seconds,"
                " status=excluded.status, updated_at=excluded.updated_at",
                (file_id, content_hash, max_tier, raw_tokens, seconds, status, time.time()),
            )

    def file_hash(self, file_id: str) -> str | None:
        row = self._meta.execute(
            "SELECT content_hash FROM files WHERE file_id = ? AND status = 'done'", (file_id,)
        ).fetchone()
        return row[0] if row else None

    def raw_tokens(self, file_id: str) -> int:
        row = self._meta.execute("SELECT raw_tokens FROM files WHERE file_id = ?", (file_id,)).fetchone()
        return int(row[0]) if row else 0

    def delete_file(self, file_id: str) -> None:
        self.delete_file_chunks(file_id)
        with self._meta:
            self._meta.execute("DELETE FROM files WHERE file_id = ?", (file_id,))

    def get_delta_link(self, folder_id: str) -> str | None:
        row = self._meta.execute("SELECT delta_link FROM sync_state WHERE folder_id = ?", (folder_id,)).fetchone()
        return row[0] if row else None

    def set_delta_link(self, folder_id: str, link: str | None) -> None:
        with self._meta:
            self._meta.execute(
                "INSERT INTO sync_state (folder_id, delta_link, last_synced) VALUES (?,?,?)"
                " ON CONFLICT(folder_id) DO UPDATE SET delta_link=excluded.delta_link, last_synced=excluded.last_synced",
                (folder_id, link, time.time()),
            )

    def record_usage(self, returned: int, raw: int, saved: int) -> None:
        with self._meta:
            self._meta.execute(
                "INSERT INTO usage (at, tokens_returned, tokens_raw_estimate, tokens_saved_estimate) VALUES (?,?,?,?)",
                (time.time(), returned, raw, saved),
            )

    def stats(self) -> dict:
        """Counts and timings only. Never returns content."""
        chunks = {}
        for tier in TIERS:
            if (self.dir / f"tier{tier}.db").exists() or tier in self._conns:
                chunks[tier] = self._db(tier).execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            else:
                chunks[tier] = 0
        files, seconds = self._meta.execute(
            "SELECT COUNT(*), COALESCE(SUM(seconds), 0) FROM files WHERE status = 'done'"
        ).fetchone()
        by_tier = dict(self._meta.execute(
            "SELECT max_tier, COUNT(*) FROM files WHERE status = 'done' GROUP BY max_tier").fetchall())
        usage = self._meta.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens_returned),0), COALESCE(SUM(tokens_raw_estimate),0),"
            " COALESCE(SUM(tokens_saved_estimate),0) FROM usage").fetchone()
        return {
            "files": files,
            "files_by_tier": {t: by_tier.get(t, 0) for t in TIERS},
            "chunks_by_tier": chunks,
            "processing_seconds": seconds,
            "queries": usage[0],
            "tokens_returned": usage[1],
            "tokens_raw_estimate": usage[2],
            "tokens_saved_estimate": usage[3],
        }

    def wipe(self) -> None:
        """Delete every database file and every key. Used by ``forget --all``."""
        self.close()
        for p in self.dir.glob("*.db*"):
            p.unlink(missing_ok=True)
        for tier in TIERS:
            self._secrets.delete(f"tier{tier}-key")
        self._aes.clear()
