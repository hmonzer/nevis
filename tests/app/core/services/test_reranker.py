"""Tests for reranker service."""
from uuid import uuid4

import pytest
from sentence_transformers import CrossEncoder

from src.app.core.services.reranker import CrossEncoderReranker
from src.app.core.domain.models import DocumentChunk, ChunkSearchResult


@pytest.fixture
def cross_encoder_model():
    """Create a CrossEncoder model instance for testing."""
    # Using a smaller model for faster tests - you can use BAAI/bge-reranker-v2-m3 in production
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@pytest.fixture
def reranker(cross_encoder_model):
    """Create a CrossEncoderReranker instance."""
    return CrossEncoderReranker(cross_encoder_model)


@pytest.fixture
def sample_documents():
    """Create sample document chunks for testing."""
    document_id = uuid4()

    # Utility bill - best proof of address
    utility_bill_chunk = DocumentChunk(
        id=uuid4(),
        document_id=document_id,
        chunk_index=0,
        chunk_content="""
        ELECTRICITY BILL
        Account Holder: John Doe
        Service Address: 123 Main Street, Apartment 4B, Springfield, IL 62701
        Billing Period: October 2024

        This bill serves as proof of residence at the above address.
        This utility bill can be used as proof of address for official purposes.
        """,
        embedding=[0.1] * 384  # Dummy embedding
    )

    # Passport - proof of identity, NOT proof of address
    passport_chunk = DocumentChunk(
        id=uuid4(),
        document_id=document_id,
        chunk_index=1,
        chunk_content="""
        PASSPORT
        Type: P (Regular Passport)
        Passport No.: 123456789
        Surname: DOE
        Given Names: JOHN MICHAEL
        Nationality: UNITED STATES OF AMERICA
        Date of Birth: 15 MAR 1985

        This document serves as proof of identity and citizenship.
        This passport is valid for international travel.
        """,
        embedding=[0.2] * 384  # Dummy embedding
    )

    # Driver's License - has address but weaker proof
    license_chunk = DocumentChunk(
        id=uuid4(),
        document_id=document_id,
        chunk_index=2,
        chunk_content="""
        DRIVER LICENSE
        State of Illinois
        License Number: D123-4567-8901
        Name: JOHN MICHAEL DOE
        Address: 123 Main Street, Apt 4B, Springfield, IL 62701
        Date of Birth: 03/15/1985

        This license authorizes the holder to operate motor vehicles.
        """,
        embedding=[0.3] * 384  # Dummy embedding
    )

    return {
        "utility_bill": utility_bill_chunk,
        "passport": passport_chunk,
        "license": license_chunk
    }


@pytest.mark.asyncio
async def test_rerank_proof_of_address(reranker, sample_documents):
    """Test that reranker ranks utility bill highest for 'proof of address' query."""
    # Arrange - Create initial search results with equal scores (simulating retrieval)
    initial_results = [
        ChunkSearchResult(chunk=sample_documents["passport"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["utility_bill"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["license"], score=0.7),
    ]

    # Act - Rerank based on query
    reranked_results = await reranker.rerank("proof of address", initial_results)

    # Assert - Utility bill should rank highest
    assert len(reranked_results) == 3
    assert reranked_results[0].chunk.id == sample_documents["utility_bill"].id, (
        "Utility bill should rank highest for 'proof of address' query"
    )

    # Verify scores are in descending order
    for i in range(len(reranked_results) - 1):
        assert reranked_results[i].score >= reranked_results[i + 1].score, (
            "Reranked results should be ordered by score descending"
        )
        # Calculate score differences
    top_score = reranked_results[0].score
    second_score = reranked_results[1].score
    score_gap = top_score - second_score

    # The gap between top result and second should be meaningful
    assert score_gap > 0.1, (
        f"Reranking should create a meaningful score gap. Got {score_gap:.4f}"
    )

@pytest.mark.asyncio
async def test_rerank_with_top_k(reranker, sample_documents):
    """Test that top_k parameter limits the number of returned results."""
    # Arrange
    initial_results = [
        ChunkSearchResult(chunk=sample_documents["utility_bill"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["passport"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["license"], score=0.7),
    ]

    # Act - Rerank with top_k=2
    reranked_results = await reranker.rerank("proof of address", initial_results, top_k=2)

    # Assert - Should only return 2 results
    assert len(reranked_results) == 2, "Should return only top_k results"


@pytest.mark.asyncio
async def test_rerank_empty_query_raises_error(reranker, sample_documents):
    """Test that empty query raises ValueError."""
    initial_results = [
        ChunkSearchResult(chunk=sample_documents["utility_bill"], score=0.7),
    ]

    with pytest.raises(ValueError, match="Query cannot be empty"):
        await reranker.rerank("", initial_results)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        await reranker.rerank("   ", initial_results)


@pytest.mark.asyncio
async def test_rerank_empty_results_raises_error(reranker):
    """Test that empty results list raises ValueError."""
    with pytest.raises(ValueError, match="Results list cannot be empty"):
        await reranker.rerank("proof of address", [])


@pytest.mark.asyncio
async def test_rerank_maintains_chunk_data(reranker, sample_documents):
    """Test that reranking preserves all chunk data, only updating scores."""
    # Arrange
    initial_results = [
        ChunkSearchResult(chunk=sample_documents["utility_bill"], score=0.5),
    ]

    original_chunk = initial_results[0].chunk

    # Act
    reranked_results = await reranker.rerank("proof of address", initial_results)

    # Assert - Chunk data should be preserved
    reranked_chunk = reranked_results[0].chunk
    assert reranked_chunk.id == original_chunk.id
    assert reranked_chunk.document_id == original_chunk.document_id
    assert reranked_chunk.chunk_index == original_chunk.chunk_index
    assert reranked_chunk.chunk_content == original_chunk.chunk_content

    # Only score should change
    assert reranked_results[0].score != initial_results[0].score


@pytest.mark.asyncio
async def test_rerank_different_queries_produce_different_rankings(reranker, sample_documents):
    """Test that different queries produce different rankings."""
    initial_results = [
        ChunkSearchResult(chunk=sample_documents["utility_bill"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["passport"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["license"], score=0.7),
    ]

    # Query 1: proof of address - should rank utility bill highest
    results_address = await reranker.rerank("proof of address", initial_results)

    # Query 2: proof of identity - should rank passport highest
    results_identity = await reranker.rerank("proof of identity", initial_results)

    # Assert - Different queries should produce different top results
    assert results_address[0].chunk.id == sample_documents["utility_bill"].id, (
        "Proof of address query should rank utility bill highest"
    )

    assert results_identity[0].chunk.id == sample_documents["passport"].id, (
        "Proof of identity query should rank passport highest"
    )
