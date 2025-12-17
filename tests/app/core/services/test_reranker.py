"""Tests for reranker service.

Uses session-scoped reranker_service fixture from conftest.py to avoid
reloading the ML model for each test.
"""
from uuid import uuid4

import pytest

from src.app.core.domain.models import DocumentChunk, ChunkSearchResult
from src.app.core.services.reranker import RankedResult


# Content extractor for ChunkSearchResult - used across all tests
def chunk_content_extractor(result: ChunkSearchResult) -> str:
    """Extract chunk content for reranking."""
    return result.chunk.chunk_content


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
async def test_rerank_proof_of_address(reranker_service, sample_documents):
    """Test that reranker ranks utility bill highest for 'proof of address' query."""
    # Arrange - Create initial search results with equal scores (simulating retrieval)
    initial_results = [
        ChunkSearchResult(chunk=sample_documents["passport"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["utility_bill"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["license"], score=0.7),
    ]

    # Act - Rerank based on query
    ranked = await reranker_service.rerank(
        query="proof of address",
        items=initial_results,
        content_extractor=chunk_content_extractor,
    )

    # Assert - Returns RankedResult objects
    assert len(ranked) == 3
    assert all(isinstance(r, RankedResult) for r in ranked)

    # Utility bill should rank highest
    assert ranked[0].item.chunk.id == sample_documents["utility_bill"].id, (
        "Utility bill should rank highest for 'proof of address' query"
    )

    # Verify scores are in descending order
    for i in range(len(ranked) - 1):
        assert ranked[i].score >= ranked[i + 1].score, (
            "Reranked results should be ordered by score descending"
        )

    # Calculate score differences
    top_score = ranked[0].score
    second_score = ranked[1].score
    score_gap = top_score - second_score

    # The gap between top result and second should be meaningful
    assert score_gap > 0.1, (
        f"Reranking should create a meaningful score gap. Got {score_gap:.4f}"
    )


@pytest.mark.asyncio
async def test_rerank_with_top_k(reranker_service, sample_documents):
    """Test that top_k parameter limits the number of returned results."""
    # Arrange
    initial_results = [
        ChunkSearchResult(chunk=sample_documents["utility_bill"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["passport"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["license"], score=0.7),
    ]

    # Act - Rerank with top_k=2
    ranked = await reranker_service.rerank(
        query="proof of address",
        items=initial_results,
        content_extractor=chunk_content_extractor,
        top_k=2,
    )

    # Assert - Should only return 2 results
    assert len(ranked) == 2, "Should return only top_k results"


@pytest.mark.asyncio
async def test_rerank_empty_query_raises_error(reranker_service, sample_documents):
    """Test that empty query raises ValueError."""
    initial_results = [
        ChunkSearchResult(chunk=sample_documents["utility_bill"], score=0.7),
    ]

    with pytest.raises(ValueError, match="Query cannot be empty"):
        await reranker_service.rerank(
            query="",
            items=initial_results,
            content_extractor=chunk_content_extractor,
        )

    with pytest.raises(ValueError, match="Query cannot be empty"):
        await reranker_service.rerank(
            query="   ",
            items=initial_results,
            content_extractor=chunk_content_extractor,
        )


@pytest.mark.asyncio
async def test_rerank_empty_results_raises_error(reranker_service):
    """Test that empty results list raises ValueError."""
    with pytest.raises(ValueError, match="Items list cannot be empty"):
        await reranker_service.rerank(
            query="proof of address",
            items=[],
            content_extractor=chunk_content_extractor,
        )


@pytest.mark.asyncio
async def test_rerank_maintains_chunk_data(reranker_service, sample_documents):
    """Test that reranking preserves all chunk data, only updating scores."""
    # Arrange
    initial_results = [
        ChunkSearchResult(chunk=sample_documents["utility_bill"], score=0.5),
    ]

    original_chunk = initial_results[0].chunk

    # Act
    ranked = await reranker_service.rerank(
        query="proof of address",
        items=initial_results,
        content_extractor=chunk_content_extractor,
    )

    # Assert - Original item is preserved in RankedResult
    assert ranked[0].item.chunk.id == original_chunk.id
    assert ranked[0].item.chunk.document_id == original_chunk.document_id
    assert ranked[0].item.chunk.chunk_index == original_chunk.chunk_index
    assert ranked[0].item.chunk.chunk_content == original_chunk.chunk_content

    # Score should be the CrossEncoder score, not the original
    assert ranked[0].score != initial_results[0].score


@pytest.mark.asyncio
async def test_rerank_different_queries_produce_different_rankings(reranker_service, sample_documents):
    """Test that different queries produce different rankings."""
    initial_results = [
        ChunkSearchResult(chunk=sample_documents["utility_bill"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["passport"], score=0.7),
        ChunkSearchResult(chunk=sample_documents["license"], score=0.7),
    ]

    # Query 1: proof of address - should rank utility bill highest
    results_address = await reranker_service.rerank(
        query="proof of address",
        items=initial_results,
        content_extractor=chunk_content_extractor,
    )

    # Query 2: proof of identity - should rank passport highest
    results_identity = await reranker_service.rerank(
        query="proof of identity",
        items=initial_results,
        content_extractor=chunk_content_extractor,
    )

    # Assert - Different queries should produce different top results
    assert results_address[0].item.chunk.id == sample_documents["utility_bill"].id, (
        "Proof of address query should rank utility bill highest"
    )

    assert results_identity[0].item.chunk.id == sample_documents["passport"].id, (
        "Proof of identity query should rank passport highest"
    )


# =============================================================================
# Generic Reranker Tests - Test with different item types
# =============================================================================


@pytest.mark.asyncio
async def test_rerank_with_simple_strings(reranker_service):
    """Test that reranker works with simple string items."""
    # Arrange - Simple string items
    items = [
        "The cat sat on the mat",
        "Dogs are loyal pets",
        "Python is a programming language",
    ]

    # Act - Rerank strings directly
    ranked = await reranker_service.rerank(
        query="pets and animals",
        items=items,
        content_extractor=lambda x: x,  # Identity function for strings
    )

    # Assert - Should return RankedResult[str]
    assert len(ranked) == 3
    assert all(isinstance(r.item, str) for r in ranked)
    # Cat and dog items should score higher than programming
    assert ranked[2].item == "Python is a programming language", (
        "Programming language should rank lowest for pets query"
    )


@pytest.mark.asyncio
async def test_rerank_with_dict_items(reranker_service):
    """Test that reranker works with dictionary items."""
    # Arrange - Dict items
    items = [
        {"id": 1, "title": "Tax Return 2024", "content": "Annual tax filing for fiscal year 2024"},
        {"id": 2, "title": "Meeting Notes", "content": "Quarterly review meeting discussion"},
        {"id": 3, "title": "Tax Deductions", "content": "List of applicable tax deductions"},
    ]

    # Act - Rerank dicts with custom extractor
    ranked = await reranker_service.rerank(
        query="tax documents",
        items=items,
        content_extractor=lambda x: f"{x['title']}. {x['content']}",
    )

    # Assert - Tax-related items should rank higher
    assert len(ranked) == 3
    top_two_ids = {ranked[0].item["id"], ranked[1].item["id"]}
    assert top_two_ids == {1, 3}, "Tax-related documents should rank in top 2"


# =============================================================================
# Client Entity Relevance Tests
# =============================================================================
# These tests validate whether CrossEncoder can distinguish between:
# 1. Queries seeking a client entity (e.g., "John Doe") - client record should score HIGH
# 2. Queries seeking specific information (e.g., "John Doe's tax record") - client record should score LOW


@pytest.fixture
def client_and_document_chunks():
    """Create client description and related document chunks for testing."""
    document_id = uuid4()

    # Client record description (as stored in client entity)
    client_description = DocumentChunk(
        id=uuid4(),
        document_id=document_id,
        chunk_index=0,
        chunk_content="""
        John Doe
        Email: john.doe@email.com
        Senior Software Engineer at Tech Corp.
        Specializes in wealth management and retirement planning.
        High net worth individual with diverse investment portfolio.
        """,
        embedding=[0.1] * 384
    )

    # Tax document for John Doe
    tax_document = DocumentChunk(
        id=uuid4(),
        document_id=document_id,
        chunk_index=1,
        chunk_content="""
        TAX RETURN 2024
        Taxpayer: John Doe
        Filing Status: Single
        Total Income: $450,000
        Federal Tax Withheld: $125,000
        State Tax Withheld: $35,000
        Adjusted Gross Income: $425,000
        Taxable Income: $380,000
        Total Tax Liability: $115,000
        Refund Amount: $10,000
        """,
        embedding=[0.2] * 384
    )

    # Investment portfolio document
    investment_document = DocumentChunk(
        id=uuid4(),
        document_id=document_id,
        chunk_index=2,
        chunk_content="""
        INVESTMENT PORTFOLIO STATEMENT Q4 2024
        Account Holder: John Doe
        Total Portfolio Value: $2,500,000
        Asset Allocation:
        - Equities: 60% ($1,500,000)
        - Fixed Income: 25% ($625,000)
        - Alternative Investments: 15% ($375,000)
        YTD Performance: +12.5%
        """,
        embedding=[0.3] * 384
    )

    return {
        "client": client_description,
        "tax_document": tax_document,
        "investment_document": investment_document,
    }


@pytest.mark.asyncio
async def test_client_query_scores_client_record_high(reranker_service, client_and_document_chunks):
    """
    Test that a simple client name query scores the client record HIGH.

    When user searches "John Doe", they likely want to find the client entity,
    not specific documents. The client record should score higher than document chunks.
    """
    initial_results = [
        ChunkSearchResult(chunk=client_and_document_chunks["client"], score=0.7),
        ChunkSearchResult(chunk=client_and_document_chunks["tax_document"], score=0.7),
        ChunkSearchResult(chunk=client_and_document_chunks["investment_document"], score=0.7),
    ]

    # Query: Simple client name lookup
    ranked = await reranker_service.rerank(
        query="John Doe",
        items=initial_results,
        content_extractor=chunk_content_extractor,
    )

    # Client record should rank highest for a name query
    assert ranked[0].item.chunk.id == client_and_document_chunks["client"].id, (
        "Client record should rank highest for simple name query"
    )

    # Client score should be notably higher
    client_score = ranked[0].score
    second_score = ranked[1].score
    print(f"Name query 'John Doe': client={client_score:.4f}, second={second_score:.4f}")

    assert client_score > second_score, (
        f"Client should score higher than documents for name query. "
        f"Client: {client_score:.4f}, Second: {second_score:.4f}"
    )


@pytest.mark.asyncio
async def test_specific_document_query_scores_client_record_low(reranker_service, client_and_document_chunks):
    """
    Test that a specific document query scores the client record LOW.

    When user searches "John Doe's tax record for 2024", they want the tax document,
    not the client entity. The client record should score lower than the relevant document.
    """
    initial_results = [
        ChunkSearchResult(chunk=client_and_document_chunks["client"], score=0.7),
        ChunkSearchResult(chunk=client_and_document_chunks["tax_document"], score=0.7),
        ChunkSearchResult(chunk=client_and_document_chunks["investment_document"], score=0.7),
    ]

    # Query: Specific document request
    ranked = await reranker_service.rerank(
        query="John Doe's tax record for 2024",
        items=initial_results,
        content_extractor=chunk_content_extractor,
    )

    # Tax document should rank highest for this specific query
    assert ranked[0].item.chunk.id == client_and_document_chunks["tax_document"].id, (
        "Tax document should rank highest for tax-specific query"
    )

    # Find client score position and value
    client_result = next(r for r in ranked if r.item.chunk.id == client_and_document_chunks["client"].id)
    tax_result = ranked[0]

    print(f"Tax query: tax_doc={tax_result.score:.4f}, client={client_result.score:.4f}")

    assert tax_result.score > client_result.score, (
        f"Tax document should score higher than client for specific query. "
        f"Tax: {tax_result.score:.4f}, Client: {client_result.score:.4f}"
    )


@pytest.mark.asyncio
async def test_investment_query_scores_relevant_document_high(reranker_service, client_and_document_chunks):
    """
    Test that investment-specific query ranks investment document highest.
    """
    initial_results = [
        ChunkSearchResult(chunk=client_and_document_chunks["client"], score=0.7),
        ChunkSearchResult(chunk=client_and_document_chunks["tax_document"], score=0.7),
        ChunkSearchResult(chunk=client_and_document_chunks["investment_document"], score=0.7),
    ]

    # Query: Investment-specific request
    ranked = await reranker_service.rerank(
        query="John Doe portfolio performance 2024",
        items=initial_results,
        content_extractor=chunk_content_extractor,
    )

    # Investment document should rank highest
    assert ranked[0].item.chunk.id == client_and_document_chunks["investment_document"].id, (
        "Investment document should rank highest for portfolio query"
    )

    # Client should rank lower than the relevant document
    client_result = next(r for r in ranked if r.item.chunk.id == client_and_document_chunks["client"].id)
    investment_result = ranked[0]

    print(f"Investment query: investment={investment_result.score:.4f}, client={client_result.score:.4f}")

    assert investment_result.score > client_result.score, (
        f"Investment document should score higher than client. "
        f"Investment: {investment_result.score:.4f}, Client: {client_result.score:.4f}"
    )


@pytest.mark.asyncio
async def test_cross_encoder_score_ranges_for_client_vs_document_queries(reranker_service, client_and_document_chunks):
    """
    Exploratory test to understand CrossEncoder score distributions.

    This test prints score comparisons to help determine appropriate thresholds
    for filtering client results based on query intent.
    """
    initial_results = [
        ChunkSearchResult(chunk=client_and_document_chunks["client"], score=0.7),
        ChunkSearchResult(chunk=client_and_document_chunks["tax_document"], score=0.7),
        ChunkSearchResult(chunk=client_and_document_chunks["investment_document"], score=0.7),
    ]

    queries = [
        # Client-seeking queries (expect HIGH client scores)
        ("John Doe", "client_lookup"),
        ("Find client John Doe", "client_lookup"),
        ("Who is John Doe", "client_lookup"),

        # Document-seeking queries (expect LOW client scores)
        ("John Doe's tax return 2024", "document_lookup"),
        ("John Doe total income", "document_lookup"),
        ("John Doe investment portfolio value", "document_lookup"),
        ("What is John Doe's taxable income", "document_lookup"),
    ]

    print("\n" + "=" * 80)
    print("CrossEncoder Score Analysis: Client vs Document Queries")
    print("=" * 80)

    client_lookup_scores = []
    document_lookup_scores = []

    for query, query_type in queries:
        ranked = await reranker_service.rerank(
            query=query,
            items=initial_results,
            content_extractor=chunk_content_extractor,
        )

        client_result = next(r for r in ranked if r.item.chunk.id == client_and_document_chunks["client"].id)
        client_score = client_result.score

        if query_type == "client_lookup":
            client_lookup_scores.append(client_score)
        else:
            document_lookup_scores.append(client_score)

        print(f"  {query_type:15} | Query: '{query[:40]:<40}' | Client score: {client_score:+.4f}")

    print("-" * 80)
    print(f"Client lookup queries - Avg client score: {sum(client_lookup_scores)/len(client_lookup_scores):+.4f}")
    print(f"Document lookup queries - Avg client score: {sum(document_lookup_scores)/len(document_lookup_scores):+.4f}")
    print("=" * 80)

    # Key assertion: client scores should be meaningfully higher for client-seeking queries
    avg_client_lookup = sum(client_lookup_scores) / len(client_lookup_scores)
    avg_document_lookup = sum(document_lookup_scores) / len(document_lookup_scores)

    assert avg_client_lookup > avg_document_lookup, (
        f"Client-seeking queries should score client higher on average. "
        f"Client lookup avg: {avg_client_lookup:.4f}, Document lookup avg: {avg_document_lookup:.4f}"
    )
