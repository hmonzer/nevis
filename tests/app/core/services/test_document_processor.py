"""Tests for document processor."""
from uuid import uuid4

import pytest
from src.app.core.services.document_processor import DocumentProcessor, ProcessingResult
from src.app.core.services.summarization import SummarizationService


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def document_processor(chunking_service, embedding_service):
    """Create a document processor without summarization."""
    return DocumentProcessor(
        chunking_strategy=chunking_service,
        embedding_service=embedding_service,
    )


class MockSummarizationService(SummarizationService):
    """Mock summarization service for testing."""

    def __init__(self, summary_text: str = "Mock summary for testing."):
        self.summary_text = summary_text
        self.call_count = 0

    async def summarize(self, content: str) -> str:
        self.call_count += 1
        return self.summary_text


@pytest.fixture
def mock_summarization_service():
    """Create a mock summarization service."""
    return MockSummarizationService(
        summary_text="This document discusses programming concepts and weather patterns."
    )


@pytest.fixture
def document_processor_with_summary(chunking_service, embedding_service, mock_summarization_service):
    """Create a document processor with summarization service."""
    return DocumentProcessor(
        chunking_strategy=chunking_service,
        embedding_service=embedding_service,
        summarization_service=mock_summarization_service,
    )


# =============================================================================
# Tests
# =============================================================================


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
    result = await document_processor.process_text(document_id, content)

    # Assert - Verify result is ProcessingResult
    assert isinstance(result, ProcessingResult)

    # Assert - Verify chunks were created
    assert len(result.chunks) > 0

    for chunk in result.chunks:
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
    # Long content to create multiple chunks (need to exceed chunk_size tokens)
    content = "This is a sentence that will be repeated many times to create chunks. " * 50

    # Act
    result = await document_processor.process_text(document_id, content)

    # Assert
    assert len(result.chunks) > 1

    # Verify indices are sequential starting from 0
    indices = sorted([chunk.chunk_index for chunk in result.chunks])
    assert indices == list(range(len(result.chunks)))


@pytest.mark.asyncio
async def test_process_text_empty_content(document_processor):
    """Test processing empty content returns empty list."""
    # Arrange
    document_id = uuid4()

    # Act
    result = await document_processor.process_text(document_id, "")

    # Assert - No chunks should be created
    assert len(result.chunks) == 0
    assert result.summary is None


@pytest.mark.asyncio
async def test_process_text_embeddings_are_different_for_different_chunks(document_processor):
    """Test that different chunks get different embeddings."""
    import numpy as np

    # Arrange
    document_id = uuid4()

    # Content with distinctly different sections - needs to exceed chunk_size tokens (256) to split
    # Generate enough content to create at least 2 chunks
    content = """
    Python is a high-level programming language created by Guido van Rossum in the late 1980s.
    It is widely used for web development, data science, machine learning, and artificial intelligence.
    Python has a simple and readable syntax that makes it easy for beginners to learn programming.
    The language supports multiple programming paradigms including procedural, object-oriented, and functional.
    Python has a large standard library and an active community of developers around the world.
    The Python Package Index contains thousands of third-party modules for various applications.
    Django and Flask are popular web frameworks built with Python for creating web applications.
    NumPy, Pandas, and Matplotlib are essential libraries for data science and scientific computing.
    TensorFlow and PyTorch are leading frameworks for deep learning and neural network development.
    Python is also used extensively in automation, scripting, and system administration tasks.
    """ + """
    The weather today is sunny and warm with clear blue skies stretching overhead across the horizon.
    People enjoy going to the beach in summer to swim in the ocean and relax on the sandy shores.
    Sunscreen is very important to protect skin from harmful ultraviolet rays from the sun.
    Many families plan their annual vacations around popular beach destinations around the world.
    The ocean waves provide a calming and soothing sound that many people find therapeutic and relaxing.
    Surfing has become an increasingly popular water sport enjoyed by millions of enthusiasts globally.
    Beach volleyball and sandcastle building are favorite activities for children and adults alike.
    Coastal ecosystems support diverse marine life including fish, dolphins, sea turtles, and seabirds.
    Climate change and rising sea levels pose significant threats to coastal communities worldwide.
    Marine conservation efforts aim to protect ocean habitats and endangered species from human impact.
    """

    # Act
    result = await document_processor.process_text(document_id, content)

    # Assert
    assert len(result.chunks) >= 2, f"Expected at least 2 chunks but got {len(result.chunks)}"

    # Get embeddings for first two chunks
    emb1 = np.array(result.chunks[0].embedding)
    emb2 = np.array(result.chunks[1].embedding)

    # They should not be identical
    assert not np.allclose(emb1, emb2)


@pytest.mark.asyncio
async def test_process_text_with_summarization_service(
    document_processor_with_summary,
    mock_summarization_service,
):
    """Test that process_text generates summary when summarization service is provided."""
    # Arrange
    document_id = uuid4()
    content = """
    This is a test document about financial planning.
    It covers investment strategies and retirement goals.
    The document should be summarized for wealth management purposes.
    """

    # Act
    result = await document_processor_with_summary.process_text(document_id, content)

    # Assert - Verify result structure
    assert isinstance(result, ProcessingResult)
    assert len(result.chunks) > 0

    # Assert - Verify summary was generated
    assert result.summary is not None
    assert result.summary == mock_summarization_service.summary_text

    # Assert - Verify summarization service was called exactly once
    assert mock_summarization_service.call_count == 1


@pytest.mark.asyncio
async def test_process_text_without_summarization_returns_none_summary(document_processor):
    """Test that process_text returns None summary when no summarization service."""
    # Arrange
    document_id = uuid4()
    content = "This is a test document."

    # Act
    result = await document_processor.process_text(document_id, content)

    # Assert
    assert isinstance(result, ProcessingResult)
    assert result.summary is None
    assert len(result.chunks) > 0
