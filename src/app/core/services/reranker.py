"""Reranking service interface and implementations for reordering search results."""
import asyncio
import logging
from abc import ABC, abstractmethod

from sentence_transformers import CrossEncoder

from src.app.core.domain.models import ChunkSearchResult

logger = logging.getLogger(__name__)


class RerankerService(ABC):
    """
    Abstract base class for reranking search results.

    Rerankers take an initial set of search results and reorder them based on
    more sophisticated relevance scoring, typically using cross-encoder models
    that consider the query-document interaction.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[ChunkSearchResult],
        top_k: int | None = None
    ) -> list[ChunkSearchResult]:
        """
        Rerank search results based on query relevance.

        Args:
            query: The search query string
            results: List of initial search results to rerank
            top_k: Optional number of top results to return. If None, returns all reranked results.

        Returns:
            List of ChunkSearchResult objects reranked by relevance score (descending)

        Raises:
            ValueError: If query is empty or results list is empty
        """
        pass


class CrossEncoderReranker(RerankerService):
    """
    Reranker using CrossEncoder models from sentence-transformers.

    CrossEncoders directly score query-document pairs, providing more accurate
    relevance scores than bi-encoder similarity, at the cost of being slower.
    This makes them ideal for reranking a smaller set of candidates.
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
        results: list[ChunkSearchResult],
        top_k: int | None = None
    ) -> list[ChunkSearchResult]:
        """
        Rerank search results using CrossEncoder scores.

        The CrossEncoder model scores each (query, document) pair directly,
        providing more accurate relevance scores than cosine similarity.

        Args:
            query: The search query string
            results: List of initial search results to rerank
            top_k: Optional number of top results to return. If None, returns all.

        Returns:
            List of ChunkSearchResult objects reranked by CrossEncoder scores (descending)

        Raises:
            ValueError: If query is empty or results list is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not results:
            raise ValueError("Results list cannot be empty")

        logger.info(
            "Reranking %d results for query: '%s' (top_k=%s)",
            len(results),
            query[:100],
            top_k
        )

        # Prepare query-document pairs for the cross-encoder
        pairs = [(query, result.chunk.chunk_content) for result in results]

        # Score all pairs using the cross-encoder
        # Run in thread pool to avoid blocking the event loop
        scores = await asyncio.to_thread(
            self.model.predict,
            pairs,
            convert_to_numpy=True
        )

        # Create new ChunkSearchResult objects with updated scores
        reranked_results = [
            ChunkSearchResult(
                chunk=result.chunk,
                score=float(score)
            )
            for result, score in zip(results, scores)
        ]

        # Sort by score descending (highest relevance first)
        reranked_results.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k limit if specified
        if top_k is not None:
            reranked_results = reranked_results[:top_k]

        logger.info(
            "Reranking complete. Top score: %.4f, Bottom score: %.4f",
            reranked_results[0].score if reranked_results else 0.0,
            reranked_results[-1].score if reranked_results else 0.0
        )

        return reranked_results
