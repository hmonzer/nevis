"""Reranking service interface and implementations for reordering search results."""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Sequence

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RankedResult(Generic[T]):
    """
    Result from reranking containing the original item and its relevance score.

    Attributes:
        item: The original item that was reranked
        score: CrossEncoder relevance score (logits, typically -12 to +12).
               Positive scores indicate relevance, negative indicate irrelevance.
               0.0 is the decision boundary (50% relevance probability).
    """
    item: T
    score: float


class RerankerService(ABC):
    """
    Abstract base class for reranking search results.

    Rerankers take an initial set of items and reorder them based on
    more sophisticated relevance scoring, typically using cross-encoder models
    that consider the query-document interaction.

    This service is generic and can rerank any item type by providing a
    content_extractor function that extracts text from each item.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        items: Sequence[T],
        content_extractor: Callable[[T], str],
        top_k: int | None = None,
    ) -> list[RankedResult[T]]:
        """
        Rerank items based on query relevance.

        Args:
            query: The search query string
            items: Sequence of items to rerank (any type)
            content_extractor: Function to extract text content from each item
            top_k: Optional number of top results to return. If None, returns all.

        Returns:
            List of RankedResult objects sorted by score descending (most relevant first)

        Raises:
            ValueError: If query is empty or items sequence is empty
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
        items: Sequence[T],
        content_extractor: Callable[[T], str],
        top_k: int | None = None,
    ) -> list[RankedResult[T]]:
        """
        Rerank items using CrossEncoder scores.

        The CrossEncoder model scores each (query, content) pair directly,
        providing more accurate relevance scores than cosine similarity.

        Args:
            query: The search query string
            items: Sequence of items to rerank
            content_extractor: Function to extract text content from each item
            top_k: Optional number of top results to return. If None, returns all.

        Returns:
            List of RankedResult[T] objects sorted by CrossEncoder scores (descending)

        Raises:
            ValueError: If query is empty or items sequence is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not items:
            raise ValueError("Items list cannot be empty")

        logger.info(
            "Reranking %d items for query: '%s' (top_k=%s)",
            len(items),
            query[:100],
            top_k
        )

        # Prepare query-content pairs
        pairs = [(query, content_extractor(item)) for item in items]

        # Score all pairs using the cross-encoder
        # Run in thread pool to avoid blocking the event loop
        scores = await asyncio.to_thread(
            self.model.predict,
            pairs,
            convert_to_numpy=True
        )

        # Create RankedResult objects
        ranked_results = [
            RankedResult(item=item, score=float(score))
            for item, score in zip(items, scores)
        ]

        # Sort by score descending (highest relevance first)
        ranked_results.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k limit if specified
        if top_k is not None:
            ranked_results = ranked_results[:top_k]

        logger.info(
            "Reranking complete. Top score: %.4f, Bottom score: %.4f",
            ranked_results[0].score if ranked_results else 0.0,
            ranked_results[-1].score if ranked_results else 0.0
        )

        return ranked_results
