"""Document processor for chunking text and generating embeddings."""
import logging
from uuid import UUID, uuid4

from src.app.core.domain.models import DocumentChunk
from src.app.core.services.chunking import ChunkingStrategy
from src.app.core.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processor for handling document chunking and embedding generation.

    This class is responsible for:
    1. Chunking document text into smaller pieces
    2. Generating embeddings for each chunk
    3. Creating DocumentChunk domain objects

    Note: This class does NOT handle database persistence - that's the
    responsibility of the calling service.
    """

    def __init__(
        self,
        chunking_strategy: ChunkingStrategy,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize the document processor.

        Args:
            chunking_strategy: Service for text chunking
            embedding_service: Service for generating embeddings
        """
        self.chunking_strategy = chunking_strategy
        self.embedding_service = embedding_service
        logger.info(
            "Initialized DocumentProcessor with embedding dimension %d",
            embedding_service.embedding_dimension
        )

    async def process_text(self, document_id: UUID, content: str) -> list[DocumentChunk]:
        """
        Process text by chunking and generating embeddings.

        This method:
        1. Chunks the content using the chunking strategy
        2. Generates embeddings for all chunks (batched for efficiency)
        3. Creates DocumentChunk domain objects with embeddings

        Args:
            document_id: The ID of the document these chunks belong to
            content: The raw text content to be chunked and embedded

        Returns:
            List of DocumentChunk objects with embeddings. Empty list if content is empty.
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
            return []

        logger.info("Created %d chunks for document %s", len(chunk_texts), document_id)

        # Generate embeddings for all chunks at once (more efficient than one-by-one)
        # Returns list of EmbeddingVectorResult with text and embedding paired together
        embedding_results = await self.embedding_service.embed_batch(chunk_texts)

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

        return chunks
