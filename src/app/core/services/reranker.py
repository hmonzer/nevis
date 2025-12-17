"""Reranking service interface and implementations for reordering search results."""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Callable, Sequence

from sentence_transformers import CrossEncoder

from src.app.core.domain.models import ScoredResult, Score, ScoreSource

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RerankerService(ABC):
    """
    Abstract base class for reranking search results.

    Rerankers take an initial set of ScoredResults and reorder them based on
    more sophisticated relevance scoring, typically using cross-encoder models
    that consider the query-document interaction.

    This service follows the middleware pattern: same type in, same type out.
    The reranker receives ScoredResult[T] and returns ScoredResult[T] with
    updated scores, preserving the score history for full audit trail.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: Sequence[ScoredResult[T]],
        content_extractor: Callable[[T], str],
        top_k: int | None = None,
    ) -> list[ScoredResult[T]]:
        """
        Rerank results based on query relevance.

        Args:
            query: The search query string
            results: Sequence of ScoredResult[T] to rerank
            content_extractor: Function to extract text content from the item
                              inside each ScoredResult (extracts from T, not ScoredResult)
            top_k: Optional number of top results to return. If None, returns all.

        Returns:
            List of ScoredResult[T] with updated CrossEncoder scores,
            sorted by score descending (most relevant first).
            Previous scores are preserved in score_history.

        Raises:
            ValueError: If query is empty or results sequence is empty
        """
        pass


class CrossEncoderReranker(RerankerService):
    """
    Reranker using CrossEncoder models from sentence-transformers.

    CrossEncoders directly score query-document pairs, providing more accurate
    relevance scores than bi-encoder similarity, at the cost of being slower.
    This makes them ideal for reranking a smaller set of candidates.

    Uses the assign_score() method to preserve score history, enabling
    full audit trail of how scores evolved through the pipeline.
    """

    def __init__(self, model: CrossEncoder):
        """
        Initialize the CrossEncoder reranker.

        Args:
            model: Pre-configured CrossEncoder model instance
        """
        self.model = model

    async def rerank(
        self,
        query: str,
        results: Sequence[ScoredResult[T]],
        content_extractor: Callable[[T], str],
        top_k: int | None = None,
    ) -> list[ScoredResult[T]]:
        """
        Rerank results using CrossEncoder scores.

        The CrossEncoder model scores each (query, content) pair directly,
        providing more accurate relevance scores than cosine similarity.
        Previous scores are preserved in score_history via assign_score().

        Args:
            query: The search query string
            results: Sequence of ScoredResult[T] to rerank
            content_extractor: Function to extract text content from the item
            top_k: Optional number of top results to return. If None, returns all.

        Returns:
            List of ScoredResult[T] sorted by CrossEncoder scores (descending),
            with previous scores preserved in score_history.

        Raises:
            ValueError: If query is empty or results sequence is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not results:
            raise ValueError("Results list cannot be empty")

        logger.info(
            "Reranking %d items for query: '%s' (top_k=%s)",
            len(results),
            query[:100],
            top_k
        )

        # Prepare query-content pairs (extract from item, not ScoredResult)
        pairs = [(query, content_extractor(result.item)) for result in results]

        # Score all pairs using the cross-encoder
        # Run in thread pool to avoid blocking the event loop
        scores = await asyncio.to_thread(
            self.model.predict,
            pairs,
            convert_to_numpy=True
        )

        # Use assign_score() to preserve history
        reranked_results = [
            result.assign_score(Score(value=float(score), source=ScoreSource.CROSS_ENCODER))
            for result, score in zip(results, scores)
        ]

        # Sort by score descending (highest relevance first)
        reranked_results.sort(key=lambda x: x.value, reverse=True)

        # Apply top_k limit if specified
        if top_k is not None:
            reranked_results = reranked_results[:top_k]

        logger.info(
            "Reranking complete. Top score: %.4f, Bottom score: %.4f",
            reranked_results[0].value if reranked_results else 0.0,
            reranked_results[-1].value if reranked_results else 0.0
        )

        return reranked_results
