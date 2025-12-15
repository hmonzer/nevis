"""Tests for Reciprocal Rank Fusion (RRF) implementation."""

import pytest
from uuid import uuid4

from src.app.core.domain.models import ChunkSearchResult, DocumentChunk
from src.app.core.services.rrf import ReciprocalRankFusion


def create_chunk(chunk_id=None, content="test content") -> DocumentChunk:
    """Helper to create a DocumentChunk for testing."""
    return DocumentChunk(
        id=chunk_id or uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        chunk_content=content,
        embedding=[0.1] * 384,
    )


def create_result(chunk: DocumentChunk, score: float) -> ChunkSearchResult:
    """Helper to create a ChunkSearchResult for testing."""
    return ChunkSearchResult(chunk=chunk, score=score)


class TestReciprocalRankFusion:
    """Tests for ReciprocalRankFusion class."""

    def test_init_default_k(self):
        """Test default k value is 60."""
        rrf = ReciprocalRankFusion()
        assert rrf.k == 60

    def test_init_custom_k(self):
        """Test custom k value."""
        rrf = ReciprocalRankFusion(k=100)
        assert rrf.k == 100

    def test_init_negative_k_raises_error(self):
        """Test that negative k raises ValueError."""
        with pytest.raises(ValueError, match="k must be non-negative"):
            ReciprocalRankFusion(k=-1)

    def test_init_zero_k_allowed(self):
        """Test that k=0 is allowed (though not recommended)."""
        rrf = ReciprocalRankFusion(k=0)
        assert rrf.k == 0

    def test_fuse_empty_lists(self):
        """Test fusion with no input lists."""
        rrf = ReciprocalRankFusion()
        result = rrf.fuse()
        assert result == []

    def test_fuse_single_empty_list(self):
        """Test fusion with a single empty list."""
        rrf = ReciprocalRankFusion()
        result = rrf.fuse([])
        assert result == []

    def test_fuse_single_list_preserves_order(self):
        """Test that fusing a single list preserves order."""
        rrf = ReciprocalRankFusion(k=60)

        chunk1 = create_chunk()
        chunk2 = create_chunk()
        chunk3 = create_chunk()

        list1 = [
            create_result(chunk1, 0.9),
            create_result(chunk2, 0.8),
            create_result(chunk3, 0.7),
        ]

        result = rrf.fuse(list1)

        assert len(result) == 3
        assert result[0].chunk.id == chunk1.id
        assert result[1].chunk.id == chunk2.id
        assert result[2].chunk.id == chunk3.id

    def test_fuse_single_list_computes_rrf_scores(self):
        """Test RRF score computation for a single list."""
        rrf = ReciprocalRankFusion(k=60)

        chunk1 = create_chunk()
        chunk2 = create_chunk()

        list1 = [
            create_result(chunk1, 0.9),
            create_result(chunk2, 0.8),
        ]

        result = rrf.fuse(list1)

        # RRF score for rank 1: 1/(60+1) = 1/61
        # RRF score for rank 2: 1/(60+2) = 1/62
        assert abs(result[0].score - 1 / 61) < 1e-10
        assert abs(result[1].score - 1 / 62) < 1e-10

    def test_fuse_two_disjoint_lists(self):
        """Test fusion of two lists with no overlapping chunks."""
        rrf = ReciprocalRankFusion(k=60)

        chunk_a1 = create_chunk()
        chunk_a2 = create_chunk()
        chunk_b1 = create_chunk()
        chunk_b2 = create_chunk()

        list_a = [create_result(chunk_a1, 0.9), create_result(chunk_a2, 0.8)]
        list_b = [create_result(chunk_b1, 0.95), create_result(chunk_b2, 0.85)]

        result = rrf.fuse(list_a, list_b)

        # All 4 chunks should be in the result
        assert len(result) == 4

        # All rank-1 items have the same RRF score, so they come first (order may vary)
        # Then all rank-2 items
        rank1_ids = {chunk_a1.id, chunk_b1.id}
        rank2_ids = {chunk_a2.id, chunk_b2.id}

        assert result[0].chunk.id in rank1_ids
        assert result[1].chunk.id in rank1_ids
        assert result[2].chunk.id in rank2_ids
        assert result[3].chunk.id in rank2_ids

    def test_fuse_overlapping_chunks_accumulate_scores(self):
        """Test that overlapping chunks accumulate RRF scores."""
        rrf = ReciprocalRankFusion(k=60)

        # Create chunks with known IDs
        shared_chunk = create_chunk()
        chunk_a = create_chunk()
        chunk_b = create_chunk()

        # Shared chunk appears in both lists
        list_a = [
            create_result(shared_chunk, 0.9),  # rank 1
            create_result(chunk_a, 0.8),  # rank 2
        ]
        list_b = [
            create_result(shared_chunk, 0.95),  # rank 1
            create_result(chunk_b, 0.85),  # rank 2
        ]

        result = rrf.fuse(list_a, list_b)

        # 3 unique chunks
        assert len(result) == 3

        # Shared chunk should be first (appears in both lists at rank 1)
        assert result[0].chunk.id == shared_chunk.id

        # Shared chunk RRF score: 1/(60+1) + 1/(60+1) = 2/61
        expected_shared_score = 2 / 61
        assert abs(result[0].score - expected_shared_score) < 1e-10

    def test_fuse_different_ranks_for_same_chunk(self):
        """Test chunk appearing at different ranks in different lists."""
        rrf = ReciprocalRankFusion(k=60)

        shared_chunk = create_chunk()
        chunk_a = create_chunk()
        chunk_b = create_chunk()

        # Shared chunk is rank 1 in list_a, rank 2 in list_b
        list_a = [
            create_result(shared_chunk, 0.9),  # rank 1
            create_result(chunk_a, 0.8),  # rank 2
        ]
        list_b = [
            create_result(chunk_b, 0.95),  # rank 1
            create_result(shared_chunk, 0.85),  # rank 2
        ]

        result = rrf.fuse(list_a, list_b)

        # Find the shared chunk in results
        shared_result = next(r for r in result if r.chunk.id == shared_chunk.id)

        # RRF score: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        expected_score = 1 / 61 + 1 / 62
        assert abs(shared_result.score - expected_score) < 1e-10

    def test_fuse_three_lists(self):
        """Test fusion of three ranked lists."""
        rrf = ReciprocalRankFusion(k=60)

        shared_chunk = create_chunk()
        chunk_a = create_chunk()
        chunk_b = create_chunk()
        chunk_c = create_chunk()

        list_a = [create_result(shared_chunk, 0.9), create_result(chunk_a, 0.8)]
        list_b = [create_result(shared_chunk, 0.95), create_result(chunk_b, 0.85)]
        list_c = [create_result(chunk_c, 0.92), create_result(shared_chunk, 0.82)]

        result = rrf.fuse(list_a, list_b, list_c)

        # 4 unique chunks
        assert len(result) == 4

        # Shared chunk should be first (appears in all 3 lists)
        assert result[0].chunk.id == shared_chunk.id

        # RRF score: 1/61 + 1/61 + 1/62 (rank 1, 1, 2)
        expected_score = 1 / 61 + 1 / 61 + 1 / 62
        assert abs(result[0].score - expected_score) < 1e-10

    def test_fuse_with_k_zero(self):
        """Test fusion with k=0 (ranks become the only factor)."""
        rrf = ReciprocalRankFusion(k=0)

        chunk1 = create_chunk()
        chunk2 = create_chunk()

        list1 = [create_result(chunk1, 0.9), create_result(chunk2, 0.8)]

        result = rrf.fuse(list1)

        # With k=0: rank 1 -> 1/1 = 1.0, rank 2 -> 1/2 = 0.5
        assert abs(result[0].score - 1.0) < 1e-10
        assert abs(result[1].score - 0.5) < 1e-10

    def test_fuse_with_limit(self):
        """Test fuse_with_limit returns only top-k results."""
        rrf = ReciprocalRankFusion(k=60)

        chunks = [create_chunk() for _ in range(5)]
        list1 = [create_result(chunk, 0.9 - i * 0.1) for i, chunk in enumerate(chunks)]

        result = rrf.fuse_with_limit(list1, limit=3)

        assert len(result) == 3
        assert result[0].chunk.id == chunks[0].id
        assert result[1].chunk.id == chunks[1].id
        assert result[2].chunk.id == chunks[2].id

    def test_fuse_with_limit_more_than_results(self):
        """Test fuse_with_limit when limit exceeds available results."""
        rrf = ReciprocalRankFusion(k=60)

        chunks = [create_chunk() for _ in range(3)]
        list1 = [create_result(chunk, 0.9) for chunk in chunks]

        result = rrf.fuse_with_limit(list1, limit=10)

        # Should return all available results
        assert len(result) == 3

    def test_fuse_preserves_chunk_data(self):
        """Test that chunk data is preserved through fusion."""
        rrf = ReciprocalRankFusion(k=60)

        chunk = create_chunk(content="important content")
        list1 = [create_result(chunk, 0.9)]

        result = rrf.fuse(list1)

        assert result[0].chunk.chunk_content == "important content"
        assert result[0].chunk.embedding == chunk.embedding

    def test_fuse_deterministic_ordering_for_equal_scores(self):
        """Test that fusion produces consistent results for equal scores."""
        rrf = ReciprocalRankFusion(k=60)

        chunk1 = create_chunk()
        chunk2 = create_chunk()

        list1 = [create_result(chunk1, 0.9)]
        list2 = [create_result(chunk2, 0.9)]

        # Run multiple times to check consistency
        results = [rrf.fuse(list1, list2) for _ in range(10)]

        # All results should have the same ordering
        first_result_order = [r.chunk.id for r in results[0]]
        for result in results[1:]:
            assert [r.chunk.id for r in result] == first_result_order

    def test_rrf_boosts_chunks_appearing_in_multiple_lists(self):
        """Test that chunks in multiple lists rank higher than single-list chunks."""
        rrf = ReciprocalRankFusion(k=60)

        # Chunk appearing in both lists at rank 2
        shared_chunk = create_chunk()
        # Chunks appearing at rank 1 but only in one list
        top_a = create_chunk()
        top_b = create_chunk()

        list_a = [create_result(top_a, 0.99), create_result(shared_chunk, 0.5)]
        list_b = [create_result(top_b, 0.99), create_result(shared_chunk, 0.5)]

        result = rrf.fuse(list_a, list_b)

        # Shared chunk (2x rank 2): 2 * 1/62 = 2/62
        # Top chunks (1x rank 1): 1/61
        # 2/62 ≈ 0.0323, 1/61 ≈ 0.0164
        # So shared chunk should rank higher

        assert result[0].chunk.id == shared_chunk.id
