"""Local text embeddings.

Two implementations share one interface (``embed(texts) -> float32 array, L2-normalised``):

- SentenceTransformerEmbedder: the real model, run locally. It never touches the network
  during normal use. Weights are fetched only by the explicit ``fetch-model`` command,
  because sentence-transformers otherwise downloads them silently from huggingface.co.
- HashingEmbedder: a deterministic character n-gram hashing embedder with no
  dependencies. It captures spelling overlap, not meaning, so it is for tests and as a
  last-resort fallback only. Search quality with it is much lower.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Character 3- and 4-gram hashing into a fixed number of buckets."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            flat = " " + re.sub(r"\s+", " ", text.lower()) + " "
            for n in (3, 4):
                for i in range(len(flat) - n + 1):
                    h = hashlib.blake2b(flat[i : i + n].encode("utf-8"), digest_size=8).digest()
                    v = int.from_bytes(h, "little")
                    out[row, v % self.dim] += 1.0 if (v >> 63) & 1 == 0 else -1.0
        return _normalise(out)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, allow_download: bool = False) -> None:
        if not allow_download:
            # Must be set before huggingface_hub is imported, or it is ignored.
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on the user's environment
            raise RuntimeError("sentence-transformers is not installed: pip install sentence-transformers") from e
        try:
            self._model = SentenceTransformer(model)
        except Exception as e:  # pragma: no cover - needs the real library
            if not allow_download:
                raise RuntimeError(
                    f"Embedding model {model!r} is not on this machine. "
                    "Run `texti fetch-model` once to download the public model weights "
                    "(no personal data is sent)."
                ) from e
            raise
        self.name = model
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs the real library
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalise(np.asarray(vecs, dtype=np.float32))


def get_embedder(model: str = DEFAULT_MODEL, allow_download: bool = False, fallback: bool = True) -> Embedder:
    """The configured model, or the hashing fallback (with ``fallback``) when it is missing."""
    if model in ("hashing", "hashing-256"):
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(model, allow_download=allow_download)
    except RuntimeError:
        if not fallback:
            raise
        import warnings

        warnings.warn(
            "Using the hashing fallback embedder: search matches spelling, not meaning. "
            "Install sentence-transformers and run `texti fetch-model` for real search quality.",
            stacklevel=2,
        )
        return HashingEmbedder()
