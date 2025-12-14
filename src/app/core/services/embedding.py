"""Embedding service interface and implementations for generating text embeddings."""
import asyncio
import logging
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingVectorResult(BaseModel):
    """
    Result of embedding a text, containing both the original text and its vector.

    This ensures text and embedding are always paired together, preventing
    mismatches when processing multiple texts.
    """
    text: str
    embedding: list[float]

    model_config = {"frozen": True}  # Make immutable for safety


class EmbeddingService(ABC):
    """
    Abstract base class for text embedding services.

    This allows us to easily swap embedding models (e.g., SentenceTransformer, OpenAI)
    without changing the consuming code.
    """

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors produced by this service.

        Returns:
            The dimension of the embedding vectors
        """
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingVectorResult:
        """
        Generate an embedding vector for the given text.

        Args:
            text: The input text to embed

        Returns:
            EmbeddingVectorResult containing the text and its embedding vector

        Raises:
            ValueError: If text is empty or invalid
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[EmbeddingVectorResult]:
        """
        Generate embedding vectors for a batch of texts.

        This is more efficient than calling embed_text multiple times.

        Args:
            texts: List of input texts to embed

        Returns:
            List of EmbeddingVectorResult objects, one for each input text

        Raises:
            ValueError: If texts list is empty or contains invalid items
        """
        pass


class SentenceTransformerEmbedding(EmbeddingService):
    """
    Embedding service using SentenceTransformer models.

    This implementation uses the sentence-transformers library with an injected
    SentenceTransformer model instance.
    """

    def __init__(self, model: SentenceTransformer):
        """
        Initialize the SentenceTransformer embedding service.

        Args:
            model: Pre-configured SentenceTransformer model instance
        """
        self.model = model
        self._embedding_dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Initialized SentenceTransformerEmbedding with dimension %d",
            self._embedding_dimension
        )

    @property
    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors produced by this service.

        Returns:
            The dimension of the embedding vectors
        """
        return self._embedding_dimension

    async def embed_text(self, text: str) -> EmbeddingVectorResult:
        """
        Generate an embedding vector for the given text.

        Runs the synchronous model encoding in a thread pool to avoid blocking
        the event loop.

        Args:
            text: The input text to embed

        Returns:
            EmbeddingVectorResult containing the text and its embedding vector

        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not text.strip():
            logger.warning("Attempted to embed empty or whitespace-only text")
            raise ValueError("Text cannot be empty or whitespace only")

        logger.debug("Generating embedding for text of length %d", len(text))

        # Run synchronous encoding in thread pool to avoid blocking event loop
        embedding = await asyncio.to_thread(
            self.model.encode,
            text,
            convert_to_numpy=True
        )

        # Ensure it's a numpy array and convert to list
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)

        return EmbeddingVectorResult(text=text, embedding=embedding_list)

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingVectorResult]:
        """
        Generate embedding vectors for a batch of texts.

        This is more efficient than calling embed_text multiple times.
        Runs the synchronous model encoding in a thread pool to avoid blocking
        the event loop.

        Args:
            texts: List of input texts to embed

        Returns:
            List of EmbeddingVectorResult objects, one for each input text

        Raises:
            ValueError: If texts list is empty or contains invalid items
        """
        if not texts:
            logger.warning("Attempted to embed empty texts list")
            raise ValueError("Texts list cannot be empty")

        if any(not text or not text.strip() for text in texts):
            logger.warning("Attempted to embed batch containing empty text")
            raise ValueError("All texts must be non-empty and not just whitespace")

        logger.debug("Generating embeddings for batch of %d texts", len(texts))

        # Run synchronous encoding in thread pool to avoid blocking event loop
        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            convert_to_numpy=True
        )

        # Create EmbeddingVectorResult for each text-embedding pair
        results = [
            EmbeddingVectorResult(text=text, embedding=embedding.tolist())
            for text, embedding in zip(texts, embeddings)
        ]

        return results
