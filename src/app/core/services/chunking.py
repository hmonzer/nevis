"""Text chunking strategies using the Strategy Pattern."""
from abc import ABC, abstractmethod

from langchain_text_splitters import RecursiveCharacterTextSplitter


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


#TODO: Add Tests for Recursive Chunking Strategy
class RecursiveChunkingStrategy(ChunkingStrategy):
    """
    Recursive chunking strategy using langchain's RecursiveCharacterTextSplitter.

    This strategy splits text recursively by different separators (paragraphs, sentences, etc.)
    to create semantically meaningful chunks.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        separators: list[str] | None = None
    ):
        """
        Initialize the recursive chunking strategy.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting (default: ["\n\n", "\n", " ", ""])
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

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

        chunks = self._splitter.split_text(text)
        return chunks