"""Tests for DocumentChunkSearchService, particularly reranker score threshold filtering."""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.app.config import ChunkSearchSettings
from src.app.core.domain.models import (
    DocumentChunk,
    ScoredResult,
    Score,
    ScoreSource,
    SearchRequest,
)
from src.app.core.services.chunks_search_service import DocumentChunkSearchService
from src.app.core.services.embedding import EmbeddingService, EmbeddingVectorResult
from src.app.core.services.reranker import RerankerService
from src.app.core.services.rrf import ReciprocalRankFusion
from src.app.infrastructure.chunks_search_repository import ChunksRepositorySearch


def create_chunk_settings(reranker_score_threshold: float = 0.0) -> ChunkSearchSettings:
    """Create chunk search settings with custom threshold."""
    return ChunkSearchSettings(reranker_score_threshold=reranker_score_threshold)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = AsyncMock(spec=EmbeddingService)
    service.embed_query.return_value = EmbeddingVectorResult(
        text="test query",
        embedding=[0.1] * 384,
    )
    return service


@pytest.fixture
def mock_search_repository():
    """Create a mock search repository."""
    repository = AsyncMock(spec=ChunksRepositorySearch)
    return repository


@pytest.fixture
def mock_rrf():
    """Create a mock RRF instance."""
    rrf = MagicMock(spec=ReciprocalRankFusion)
    return rrf


@pytest.fixture
def mock_reranker_service():
    """Create a mock reranker service."""
    service = AsyncMock(spec=RerankerService)
    return service


def create_chunk_result(score: float, content: str = "Test content", source: ScoreSource = ScoreSource.RRF_FUSION) -> ScoredResult[DocumentChunk]:
    """Helper to create a ScoredResult[DocumentChunk] with given score."""
    return ScoredResult(
        item=DocumentChunk(
            id=uuid4(),
            document_id=uuid4(),
            chunk_index=0,
            chunk_content=content
        ),
        score=Score(value=score, source=source)
    )


def to_reranked_results(results: list[ScoredResult[DocumentChunk]]) -> list[ScoredResult[DocumentChunk]]:
    """Convert RRF scored results to cross-encoder scored results for mocking reranker output."""
    return [
        ScoredResult(
            item=r.item,
            score=Score(value=r.value, source=ScoreSource.CROSS_ENCODER),
            score_history=[r.score]  # Preserve original score in history
        )
        for r in results
    ]


# =============================================================================
# Tests for Reranker Score Threshold Filtering
# =============================================================================

@pytest.mark.asyncio
async def test_filtering_applied_when_reranker_enabled(
    mock_embedding_service,
    mock_search_repository,
    mock_rrf,
    mock_reranker_service,
):
    """Test that results below threshold are filtered when reranker is enabled."""
    # Arrange - Create service with reranker and threshold of 0.0
    service = DocumentChunkSearchService(
        embedding_service=mock_embedding_service,
        search_repository=mock_search_repository,
        rrf=mock_rrf,
        settings=create_chunk_settings(reranker_score_threshold=0.0),
        reranker_service=mock_reranker_service,
    )

    # Mock search results - mix of positive and negative scores (cross-encoder logits)
    fused_results = [
        create_chunk_result(4.5, "Highly relevant document"),
        create_chunk_result(1.2, "Relevant document"),
        create_chunk_result(-2.5, "Borderline document"),
        create_chunk_result(-8.0, "Irrelevant document"),
        create_chunk_result(-11.0, "Very irrelevant document"),
    ]

    mock_search_repository.search_by_keyword.return_value = []
    mock_search_repository.search_by_vector.return_value = fused_results
    mock_rrf.fuse.return_value = fused_results

    # Reranker returns ScoredResult objects with cross-encoder scores
    mock_reranker_service.rerank.return_value = to_reranked_results(fused_results)

    # Act
    request = SearchRequest(query="test query", top_k=10)
    results = await service.search(request)

    # Assert - Only results with score >= 0.0 should remain
    assert len(results) == 2
    assert results[0].value == 4.5
    assert results[1].value == 1.2


@pytest.mark.asyncio
async def test_filtering_not_applied_when_reranker_disabled(
    mock_embedding_service,
    mock_search_repository,
    mock_rrf,
):
    """Test that filtering is NOT applied when reranker is disabled (None)."""
    # Arrange - Create service WITHOUT reranker
    service = DocumentChunkSearchService(
        embedding_service=mock_embedding_service,
        search_repository=mock_search_repository,
        rrf=mock_rrf,
        settings=create_chunk_settings(reranker_score_threshold=0.0),
        reranker_service=None,  # No reranker
    )

    # Mock RRF results - these are RRF scores (0-1 range), not cross-encoder logits
    fused_results = [
        create_chunk_result(0.05, "Result 1"),
        create_chunk_result(0.03, "Result 2"),
        create_chunk_result(-2, "Result 3"),
        create_chunk_result(-5, "Result 4"),
    ]

    mock_search_repository.search_by_keyword.return_value = []
    mock_search_repository.search_by_vector.return_value = fused_results
    mock_rrf.fuse.return_value = fused_results

    # Act
    request = SearchRequest(query="test query", top_k=10)
    results = await service.search(request)

    # Assert - All results should be returned (no filtering without reranker)
    assert len(results) == 4


@pytest.mark.asyncio
async def test_filtering_with_negative_threshold(
    mock_embedding_service,
    mock_search_repository,
    mock_rrf,
    mock_reranker_service,
):
    """Test filtering with a permissive negative threshold (-3.0)."""
    # Arrange - Create service with negative threshold (more permissive)
    service = DocumentChunkSearchService(
        embedding_service=mock_embedding_service,
        search_repository=mock_search_repository,
        rrf=mock_rrf,
        settings=create_chunk_settings(reranker_score_threshold=-3.0),  # Include borderline results
        reranker_service=mock_reranker_service,
    )

    # Mock search results
    fused_results = [
        create_chunk_result(4.5, "Highly relevant"),
        create_chunk_result(1.2, "Relevant"),
        create_chunk_result(-2.5, "Borderline - should be included"),
        create_chunk_result(-8.0, "Irrelevant - should be filtered"),
        create_chunk_result(-11.0, "Very irrelevant - should be filtered"),
    ]

    mock_search_repository.search_by_keyword.return_value = []
    mock_search_repository.search_by_vector.return_value = fused_results
    mock_rrf.fuse.return_value = fused_results
    mock_reranker_service.rerank.return_value = to_reranked_results(fused_results)

    # Act
    request = SearchRequest(query="test query", top_k=10)
    results = await service.search(request)

    # Assert - Results with score >= -3.0 should remain
    assert len(results) == 3
    assert results[0].value == 4.5
    assert results[1].value == 1.2
    assert results[2].value == -2.5


@pytest.mark.asyncio
async def test_filtering_with_strict_positive_threshold(
    mock_embedding_service,
    mock_search_repository,
    mock_rrf,
    mock_reranker_service,
):
    """Test filtering with a strict positive threshold (2.0)."""
    # Arrange - Create service with strict threshold
    service = DocumentChunkSearchService(
        embedding_service=mock_embedding_service,
        search_repository=mock_search_repository,
        rrf=mock_rrf,
        settings=create_chunk_settings(reranker_score_threshold=2.0),  # Only highly relevant results
        reranker_service=mock_reranker_service,
    )

    # Mock search results
    fused_results = [
        create_chunk_result(4.5, "Highly relevant - included"),
        create_chunk_result(2.5, "Good - included"),
        create_chunk_result(1.8, "Decent - filtered"),
        create_chunk_result(0.5, "OK - filtered"),
        create_chunk_result(-2.0, "Not good - filtered"),
    ]

    mock_search_repository.search_by_keyword.return_value = []
    mock_search_repository.search_by_vector.return_value = fused_results
    mock_rrf.fuse.return_value = fused_results
    mock_reranker_service.rerank.return_value = to_reranked_results(fused_results)

    # Act
    request = SearchRequest(query="test query", top_k=10)
    results = await service.search(request)

    # Assert - Only results with score >= 2.0 should remain
    assert len(results) == 2
    assert results[0].value == 4.5
    assert results[1].value == 2.5


@pytest.mark.asyncio
async def test_filtering_all_results_below_threshold(
    mock_embedding_service,
    mock_search_repository,
    mock_rrf,
    mock_reranker_service,
):
    """Test that all results can be filtered out if none meet threshold."""
    # Arrange - Create service with high threshold
    service = DocumentChunkSearchService(
        embedding_service=mock_embedding_service,
        search_repository=mock_search_repository,
        rrf=mock_rrf,
        settings=create_chunk_settings(reranker_score_threshold=5.0),  # Very strict
        reranker_service=mock_reranker_service,
    )

    # Mock search results - all below threshold
    fused_results = [
        create_chunk_result(4.5, "Below threshold"),
        create_chunk_result(2.0, "Below threshold"),
        create_chunk_result(-1.0, "Below threshold"),
    ]

    mock_search_repository.search_by_keyword.return_value = []
    mock_search_repository.search_by_vector.return_value = fused_results
    mock_rrf.fuse.return_value = fused_results
    mock_reranker_service.rerank.return_value = to_reranked_results(fused_results)

    # Act
    request = SearchRequest(query="test query", top_k=10)
    results = await service.search(request)

    # Assert - No results should remain
    assert len(results) == 0


@pytest.mark.asyncio
async def test_filtering_preserves_order(
    mock_embedding_service,
    mock_search_repository,
    mock_rrf,
    mock_reranker_service,
):
    """Test that filtering preserves the score order of results."""
    # Arrange
    service = DocumentChunkSearchService(
        embedding_service=mock_embedding_service,
        search_repository=mock_search_repository,
        rrf=mock_rrf,
        settings=create_chunk_settings(reranker_score_threshold=0.0),
        reranker_service=mock_reranker_service,
    )

    # Mock results in descending score order
    fused_results = [
        create_chunk_result(8.0, "Best"),
        create_chunk_result(5.0, "Good"),
        create_chunk_result(-1.0, "Filtered"),
        create_chunk_result(2.0, "OK"),  # Out of order to test preservation
        create_chunk_result(-5.0, "Filtered"),
        create_chunk_result(0.5, "Borderline OK"),
    ]

    mock_search_repository.search_by_keyword.return_value = []
    mock_search_repository.search_by_vector.return_value = fused_results
    mock_rrf.fuse.return_value = fused_results
    mock_reranker_service.rerank.return_value = to_reranked_results(fused_results)

    # Act
    request = SearchRequest(query="test query", top_k=10)
    results = await service.search(request)

    # Assert - Order should be preserved (as returned by reranker)
    assert len(results) == 4
    assert results[0].value == 8.0
    assert results[1].value == 5.0
    assert results[2].value == 2.0
    assert results[3].value == 0.5


@pytest.mark.asyncio
async def test_filtering_with_exact_threshold_value(
    mock_embedding_service,
    mock_search_repository,
    mock_rrf,
    mock_reranker_service,
):
    """Test that results exactly at threshold are included (>= comparison)."""
    # Arrange
    service = DocumentChunkSearchService(
        embedding_service=mock_embedding_service,
        search_repository=mock_search_repository,
        rrf=mock_rrf,
        settings=create_chunk_settings(reranker_score_threshold=0.0),
        reranker_service=mock_reranker_service,
    )

    # Mock results with exact threshold value
    fused_results = [
        create_chunk_result(1.0, "Above threshold"),
        create_chunk_result(0.0, "Exactly at threshold - should be included"),
        create_chunk_result(-0.001, "Just below threshold - should be filtered"),
    ]

    mock_search_repository.search_by_keyword.return_value = []
    mock_search_repository.search_by_vector.return_value = fused_results
    mock_rrf.fuse.return_value = fused_results
    mock_reranker_service.rerank.return_value = to_reranked_results(fused_results)

    # Act
    request = SearchRequest(query="test query", top_k=10)
    results = await service.search(request)

    # Assert - Results at exactly the threshold should be included
    assert len(results) == 2
    assert results[0].value == 1.0
    assert results[1].value == 0.0


@pytest.mark.asyncio
async def test_default_chunk_settings_threshold(
    mock_embedding_service,
    mock_search_repository,
    mock_rrf,
    mock_reranker_service,
):
    """Test that default ChunkSearchSettings threshold is 2.0."""
    # Arrange - Create service with default settings
    service = DocumentChunkSearchService(
        embedding_service=mock_embedding_service,
        search_repository=mock_search_repository,
        rrf=mock_rrf,
        settings=ChunkSearchSettings(),  # Use defaults
        reranker_service=mock_reranker_service,
    )

    # Assert default value from ChunkSearchSettings
    assert service.settings.reranker_score_threshold == 2.0
