"""Document processor for chunking text and generating embeddings."""
import logging
from dataclasses import dataclass
from uuid import UUID

from src.app.core.domain.models import DocumentChunk
from src.app.core.services.chunking import ChunkingStrategy
from src.app.core.services.embedding import EmbeddingService
from src.app.core.services.summarization import SummarizationService, SummarizationError

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of document processing including chunks and optional summary."""
    chunks: list[DocumentChunk]
    summary: str | None


class DocumentProcessor:
    """
    Processor for handling document chunking and embedding generation.

    This class is responsible for:
    1. Chunking document text into smaller pieces
    2. Generating embeddings for each chunk
    3. Creating DocumentChunk domain objects
    4. Optionally generating document summaries using LLM

    Note: This class does NOT handle database persistence - that's the
    responsibility of the calling service.
    """

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy,
        embedding_service: EmbeddingService,
        summarization_service: SummarizationService | None = None,
    ):
        """
        Initialize the document processor.

        Args:
            chunking_strategy: Service for text chunking
            embedding_service: Service for generating embeddings
            summarization_service: Optional service for generating summaries
        """
        self.chunking_strategy = chunking_strategy
        self.embedding_service = embedding_service
        self.summarization_service = summarization_service

    async def process_text(self, document_id: UUID, content: str) -> ProcessingResult:
        """
        Process text by chunking, generating embeddings, and optionally summarizing.

        This method:
        1. Chunks the content using the chunking strategy
        2. Generates embeddings for all chunks (batched for efficiency)
        3. Creates DocumentChunk domain objects with embeddings
        4. Optionally generates a summary if summarization service is available

        Args:
            document_id: The ID of the document these chunks belong to
            content: The raw text content to be chunked and embedded

        Returns:
            ProcessingResult containing chunks and optional summary.
        """
        logger.info(
            "Processing text for document %s, content length: %d",
            document_id,
            len(content)
        )

        # Chunk the text
        chunk_texts = self.chunking_strategy.chunk_text(content)

        if not chunk_texts:
            logger.info("No chunks created for document %s (empty content)", document_id)
            return ProcessingResult(chunks=[], summary=None)

        logger.info("Created %d chunks for document %s", len(chunk_texts), document_id)

        # Generate embeddings for all chunks at once (more efficient than one-by-one)
        # Returns list of EmbeddingVectorResult with text and embedding paired together
        embedding_results = await self.embedding_service.embed_document_batch(chunk_texts)

        # Create DocumentChunk objects with embeddings
        # No need to zip - each result already contains text and embedding paired
        chunks: list[DocumentChunk] = []
        for index, result in enumerate(embedding_results):
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=index,
                chunk_content=result.text,
                embedding=result.embedding,
            )
            chunks.append(chunk)

        logger.info(
            "Successfully processed %d chunks with embeddings for document %s",
            len(chunks),
            document_id
        )

        # Generate summary if summarization service is available
        summary = await self._generate_summary(document_id, content)

        return ProcessingResult(chunks=chunks, summary=summary)

    async def _generate_summary(self, document_id: UUID, content: str) -> str | None:
        """
        Generate a summary for the document content.

        Args:
            document_id: The ID of the document
            content: The document content to summarize

        Returns:
            The summary string, or None if summarization is unavailable or fails.
        """
        if not self.summarization_service:
            logger.debug("Summarization service not configured for document %s", document_id)
            return None

        try:
            summary = await self.summarization_service.summarize(content)
            if summary:
                logger.info("Generated summary for document %s (%d chars)", document_id, len(summary))
                return summary
            return None
        except SummarizationError as e:
            logger.warning(
                "Summarization failed for document %s: %s. Continuing without summary.",
                document_id,
                e
            )
            return None
