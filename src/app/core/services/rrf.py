"""Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists."""

from uuid import UUID
from src.app.core.domain.models import (
    DocumentChunk,
    ScoredResult,
    ScoreSource,
)


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

    The score history is preserved: if a chunk appears in multiple input lists,
    all its original scores are included in score_history.

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
        *ranked_lists: list[ScoredResult[DocumentChunk]],
    ) -> list[ScoredResult[DocumentChunk]]:
        """
        Fuse multiple ranked lists using Reciprocal Rank Fusion.

        For each chunk, the score_history is populated with all scores from
        the input lists (preserving provenance for debugging) using assign_score().

        Args:
            *ranked_lists: Variable number of ranked result lists.
                          Each list should be ordered by relevance (best first).

        Returns:
            A single fused list of ScoredResult[DocumentChunk] objects, sorted by
            RRF score (highest first). The score_history contains all original
            scores from input lists.
        """
        if not ranked_lists:
            return []

        # Track RRF scores and accumulated ScoredResults for each chunk
        rrf_scores: dict[UUID, float] = {}
        chunk_results: dict[UUID, ScoredResult[DocumentChunk]] = {}

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, start=1):
                chunk_id = result.item.id

                # Compute RRF contribution: 1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution

                if chunk_id not in chunk_results:
                    # First occurrence - store as-is
                    chunk_results[chunk_id] = result
                else:
                    # Merge scores using assign_score to preserve history
                    existing = chunk_results[chunk_id]
                    # Add each historical score from the new result
                    for hist_score in result.score_history:
                        existing = existing.assign_score(hist_score)
                    # Add the new result's current score
                    existing = existing.assign_score(result.score)
                    chunk_results[chunk_id] = existing

        # Build fused results by assigning RRF scores using assign_score
        fused_results = [
            chunk_results[chunk_id].assign_score(ScoreSource.RRF_FUSION.of(score))
            for chunk_id, score in rrf_scores.items()
        ]

        # Sort by score descending (highest RRF score first)
        fused_results.sort(key=lambda r: r.value, reverse=True)

        return fused_results

    def fuse_with_limit(
        self,
        *ranked_lists: list[ScoredResult[DocumentChunk]],
        limit: int = 10,
    ) -> list[ScoredResult[DocumentChunk]]:
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
