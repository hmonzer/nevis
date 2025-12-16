"""Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists."""

from uuid import UUID
from src.app.core.domain.models import ChunkSearchResult, DocumentChunk


class ReciprocalRankFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

    RRF is a simple yet effective method for fusing multiple ranked lists that
    doesn't require score normalization. The formula is:

        RRF_score(d) = Σ 1 / (k + rank_i(d))

    where:
        - d is a document
        - k is a constant (typically 60) that mitigates the impact of high rankings
        - rank_i(d) is the rank of document d in the i-th ranking (1-indexed)

    Reference: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank learning methods"
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF with the smoothing constant k.

        Args:
            k: Smoothing constant that reduces the impact of high rankings.
               Default is 60, which is the standard value from the original paper.
               Higher values give more weight to lower-ranked documents.
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k = k

    def fuse(
        self,
        *ranked_lists: list[ChunkSearchResult],
    ) -> list[ChunkSearchResult]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.

        Args:
            *ranked_lists: Variable number of ranked result lists.
                          Each list should be ordered by relevance (best first).

        Returns:
            A single fused list of ChunkSearchResult objects, sorted by RRF score
            (highest first). The score field contains the RRF fusion score.
        """
        if not ranked_lists:
            return []

        # Track RRF scores and best chunk representation for each unique chunk ID
        rrf_scores: dict[UUID, float] = {}
        chunk_map: dict[UUID, DocumentChunk] = {}

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, start=1):
                chunk_id = result.chunk.id

                # Compute RRF contribution: 1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank)

                # Accumulate RRF score
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution

                # Store chunk (first occurrence wins, they should be identical)
                if chunk_id not in chunk_map:
                    chunk_map[chunk_id] = result.chunk

        # Build fused results sorted by RRF score (descending)
        fused_results = [
            ChunkSearchResult(chunk=chunk_map[chunk_id], score=score)
            for chunk_id, score in rrf_scores.items()
        ]

        # Sort by score descending (highest RRF score first)
        fused_results.sort(key=lambda r: r.score, reverse=True)

        return fused_results

    def fuse_with_limit(
        self,
        *ranked_lists: list[ChunkSearchResult],
        limit: int = 10,
    ) -> list[ChunkSearchResult]:
        """
        Fuse multiple ranked lists and return top-k results.

        This is a convenience method that applies a limit after fusion.

        Args:
            *ranked_lists: Variable number of ranked result lists.
            limit: Maximum number of results to return.

        Returns:
            Top-k fused results sorted by RRF score.
        """
        fused = self.fuse(*ranked_lists)
        return fused[:limit]
