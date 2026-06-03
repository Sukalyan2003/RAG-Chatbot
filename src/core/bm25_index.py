from __future__ import annotations

"""
BM25 Sparse Index Module

Wraps ``rank_bm25.BM25Okapi`` to provide a lexical (sparse) retrieval leg that
runs alongside dense embedding similarity. Chunks are tracked by a caller-supplied
stable ``chunk_id`` so the sparse index can be fused with the dense ranking via
reciprocal rank fusion.

``BM25Okapi`` builds its statistics at construction time and has no incremental
API, so this wrapper keeps the tokenized corpus in memory and rebuilds the
underlying model lazily whenever the corpus changes.
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when dep is missing
    BM25Okapi = None
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not available; hybrid retrieval will fall back to dense-only")


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace/punctuation split used for both indexing and queries."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory BM25 index keyed by stable chunk ids.

    The corpus is held as parallel ``chunk_ids`` / ``tokenized`` lists. The
    backing ``BM25Okapi`` is rebuilt lazily on the next ``query`` after any
    mutation, so batched ``add``/``remove`` calls cost a single rebuild.
    """

    def __init__(self) -> None:
        self.chunk_ids: List[str] = []
        self.tokenized: List[List[str]] = []
        self._bm25 = None

    @property
    def available(self) -> bool:
        """Whether the BM25 backend is importable and the corpus is non-empty."""
        return BM25_AVAILABLE and bool(self.tokenized)

    def add(self, chunks: List[Tuple[str, str]]) -> None:
        """Append ``(chunk_id, text)`` pairs and invalidate the cached model."""
        if not chunks:
            return
        for chunk_id, text in chunks:
            self.chunk_ids.append(chunk_id)
            self.tokenized.append(_tokenize(text))
        self._bm25 = None

    def remove(self, chunk_ids: List[str]) -> None:
        """Drop the given chunk ids and invalidate the cached model."""
        if not chunk_ids:
            return
        drop = set(chunk_ids)
        kept = [
            (cid, toks)
            for cid, toks in zip(self.chunk_ids, self.tokenized)
            if cid not in drop
        ]
        self.chunk_ids = [cid for cid, _ in kept]
        self.tokenized = [toks for _, toks in kept]
        self._bm25 = None

    def clear(self) -> None:
        """Reset the index to empty."""
        self.chunk_ids = []
        self.tokenized = []
        self._bm25 = None

    def query(self, text: str, k: int) -> List[Tuple[str, float]]:
        """Return up to ``k`` ``(chunk_id, score)`` pairs ranked by BM25 score.

        Returns an empty list when the backend is unavailable, the corpus is
        empty, or the query has no usable tokens.
        """
        if not self.available:
            return []

        query_tokens = _tokenize(text)
        if not query_tokens:
            return []

        if self._bm25 is None:
            self._bm25 = BM25Okapi(self.tokenized)

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self.chunk_ids, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [(cid, float(score)) for cid, score in ranked[:k]]
