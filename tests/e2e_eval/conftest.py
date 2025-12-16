"""Shared fixtures for e2e evaluation tests.

Most fixtures are inherited from tests/conftest.py.
This file only contains e2e-eval-specific fixtures or aliases.
"""
import pytest_asyncio


@pytest_asyncio.fixture
def document_search_repository(test_container):
    """Alias for chunks_search_repository (used in e2e eval tests)."""
    return test_container.chunks_search_repository()