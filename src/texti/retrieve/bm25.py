"""BM25 (Best Matching 25) keyword scoring for hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Extract alphanumeric terms, lowercased."""
    return [w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text)]


def compute_bm25_scores(
    query: str,
    documents: list[str],
    doc_lengths: list[int] | None = None,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """Calculate BM25 scores for documents given a query string.

    Returns a list of float scores of the same length as documents.
    """
    if not documents:
        return []

    q_terms = tokenize(query)
    if not q_terms:
        return [0.0] * len(documents)

    doc_tokens = [tokenize(d) for d in documents]
    N = len(doc_tokens)
    if doc_lengths is not None and len(doc_lengths) == N:
        lengths = doc_lengths
    else:
        lengths = [len(dt) for dt in doc_tokens]

    avgdl = max(1.0, sum(lengths) / N)

    # Document frequencies for query terms
    df = Counter()
    for dt in doc_tokens:
        dt_set = set(dt)
        for t in q_terms:
            if t in dt_set:
                df[t] += 1

    scores = [0.0] * N
    for t in q_terms:
        n = df[t]
        if n == 0:
            continue
        # Standard Lucene/BM25 IDF formula
        idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
        for i, dt in enumerate(doc_tokens):
            tf = dt.count(t)
            if tf > 0:
                doc_len = lengths[i]
                tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avgdl)))
                scores[i] += idf * tf_norm

    return scores
