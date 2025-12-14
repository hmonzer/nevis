"""Tests for document processor."""
from uuid import uuid4

import pytest
from sentence_transformers import SentenceTransformer
from src.app.core.services.document_processor import DocumentProcessor
from src.app.core.services.embedding import SentenceTransformerEmbedding

from src.app.core.services.chunking import RecursiveChunkingStrategy


@pytest.fixture
def chunking_service():
    """Create a chunking service."""
    return RecursiveChunkingStrategy(chunk_size=100, chunk_overlap=20)


@pytest.fixture
def sentence_transformer_model():
    """Create a SentenceTransformer model instance."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def embedding_service(sentence_transformer_model):
    """Create an embedding service."""
    return SentenceTransformerEmbedding(sentence_transformer_model)


@pytest.fixture
def document_processor(chunking_service, embedding_service):
    """Create a document processor."""
    return DocumentProcessor(
        chunking_strategy=chunking_service,
        embedding_service=embedding_service,
    )


@pytest.mark.asyncio
async def test_process_text_creates_chunks_with_embeddings(document_processor):
    """Test that process_text creates chunks with embeddings."""
    # Arrange
    document_id = uuid4()
    content = """
    This is a test document with multiple sentences.
    The chunking service will split this into smaller pieces.
    Each chunk should get its own embedding vector.
    The embedding vectors should be 384-dimensional for all-MiniLM-L6-v2.
    """

    # Act - Process the text
    chunks = await document_processor.process_text(document_id, content)

    # Assert - Verify chunks were created
    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.document_id == document_id
        assert chunk.chunk_content is not None
        assert len(chunk.chunk_content) > 0
        # Verify embedding exists and has correct dimension
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384
        assert all(isinstance(val, float) for val in chunk.embedding)


@pytest.mark.asyncio
async def test_process_text_chunks_ordered_by_index(document_processor):
    """Test that chunks are created with correct sequential indices."""
    # Arrange
    document_id = uuid4()
    content = "A" * 500  # Long content to create multiple chunks

    # Act
    chunks = await document_processor.process_text(document_id, content)

    # Assert
    assert len(chunks) > 1

    # Verify indices are sequential starting from 0
    indices = sorted([chunk.chunk_index for chunk in chunks])
    assert indices == list(range(len(chunks)))


@pytest.mark.asyncio
async def test_process_text_empty_content(document_processor):
    """Test processing empty content returns empty list."""
    # Arrange
    document_id = uuid4()

    # Act
    chunks = await document_processor.process_text(document_id, "")

    # Assert - No chunks should be created
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_process_text_embeddings_are_different_for_different_chunks(document_processor):
    """Test that different chunks get different embeddings."""
    import numpy as np

    # Arrange
    document_id = uuid4()

    # Content with distinctly different sections
    content = """
    Python is a high-level programming language.
    It was created by Guido van Rossum.
    """ + "X" * 200 + """
    The weather today is sunny and warm.
    People enjoy going to the beach in summer.
    """

    # Act
    chunks = await document_processor.process_text(document_id, content)

    # Assert
    assert len(chunks) >= 2

    # Get embeddings for first two chunks
    emb1 = np.array(chunks[0].embedding)
    emb2 = np.array(chunks[1].embedding)

    # They should not be identical
    assert not np.allclose(emb1, emb2)
