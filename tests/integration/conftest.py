"""Integration test fixtures for document search and processing tests.

Most fixtures are inherited from tests/conftest.py.
This file contains integration-test-specific fixtures that override
or differ from the common fixtures.
"""
import pytest_asyncio


@pytest_asyncio.fixture
def search_repository(test_container):
    """Alias for chunks_search_repository (used in integration tests)."""
    return test_container.chunks_search_repository()


@pytest_asyncio.fixture
def search_service_no_rerank(test_container):
    """
    Document chunk search service WITHOUT reranking.

    Note: This overrides the root fixture because integration tests
    work directly with DocumentChunkSearchService, not the full SearchService.
    """
    return test_container.document_chunk_search_service_no_rerank()


@pytest_asyncio.fixture
def search_service(test_container):
    """
    Document chunk search service WITH reranking.

    Note: This overrides the root fixture because integration tests
    work directly with DocumentChunkSearchService, not the full SearchService.
    """
    return test_container.document_chunk_search_service()
