"""Text chunking strategies using the Strategy Pattern."""
from abc import ABC, abstractmethod

from langchain_text_splitters import TextSplitter


class ChunkingStrategy(ABC):
    """
    Abstract base class for text chunking strategies.

    This allows us to easily swap chunking algorithms (e.g., RecursiveChunking, SemanticChunking)
    without changing the consuming code.
    """

    @abstractmethod
    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks
        """
        pass


class RecursiveChunkingStrategy(ChunkingStrategy):
    """
    Recursive chunking strategy using langchain's TextSplitter
    with a HuggingFace tokenizer for accurate token-based splitting.

    This strategy splits text recursively by different separators (paragraphs, sentences, etc.)
    to create semantically meaningful chunks sized by tokens rather than characters.
    Using token-based chunking ensures chunks respect the embedding model's context window.
    """

    def __init__(self, splitter: TextSplitter):
        """
        Initialize the recursive chunking strategy with an injected text splitter.

        Args:
            splitter: Pre-configured TextSplitter instance (typically RecursiveCharacterTextSplitter).
                     The splitter should be created with a HuggingFace tokenizer
                     for accurate token counting.
        """
        self._splitter = splitter

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks using recursive character splitting.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        return self._splitter.split_text(text)