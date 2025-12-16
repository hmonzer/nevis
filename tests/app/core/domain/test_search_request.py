"""Tests for SearchRequest domain model validation."""
from typing import cast

import pytest
from pydantic import ValidationError

from src.app.core.domain.models import SearchRequest


class TestSearchRequestValidation:
    """Test SearchRequest model validation."""

    def test_valid_request_with_all_parameters(self):
        """Test creating a valid SearchRequest with all parameters."""
        request = SearchRequest(query="test query", top_k=10)

        assert request.query == "test query"
        assert request.top_k == 10

    def test_valid_request_with_defaults(self):
        """Test creating a valid SearchRequest with default parameters."""
        request = SearchRequest(query="test query")

        assert request.query == "test query"
        assert request.top_k == 10  # Default

    def test_query_whitespace_trimming(self):
        """Test that query strings are trimmed of leading/trailing whitespace."""
        request = SearchRequest(query="  test query  ")

        assert request.query == "test query"
        assert request.query == request.query.strip()

    def test_empty_query_raises_error(self):
        """Test that empty query string raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="")

        error = cast(ValidationError, exc_info.value)
        errors = error.errors()
        assert len(errors) > 0
        assert any(err["loc"] == ("query",) for err in errors)

    def test_whitespace_only_query_raises_error(self):
        """Test that whitespace-only query raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="   ")

        error = cast(ValidationError, exc_info.value)
        errors = error.errors()
        assert len(errors) > 0
        assert any(err["loc"] == ("query",) for err in errors)

    def test_missing_query_raises_error(self):
        """Test that missing query parameter raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest()  # type: ignore

        error = cast(ValidationError, exc_info.value)
        errors = error.errors()
        assert len(errors) > 0
        assert any(err["loc"] == ("query",) for err in errors)


class TestSearchRequestTopK:
    """Test SearchRequest top_k validation."""

    def test_top_k_minimum_valid_value(self):
        """Test that top_k=1 is valid (minimum)."""
        request = SearchRequest(query="test", top_k=1)
        assert request.top_k == 1

    def test_top_k_maximum_valid_value(self):
        """Test that top_k=100 is valid (maximum)."""
        request = SearchRequest(query="test", top_k=100)
        assert request.top_k == 100

    def test_top_k_zero_raises_error(self):
        """Test that top_k=0 raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", top_k=0)

        error = cast(ValidationError, exc_info.value)
        errors = error.errors()
        assert len(errors) > 0
        assert any(err["loc"] == ("top_k",) for err in errors)

    def test_top_k_negative_raises_error(self):
        """Test that negative top_k raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", top_k=-1)

        error = cast(ValidationError, exc_info.value)
        errors = error.errors()
        assert len(errors) > 0
        assert any(err["loc"] == ("top_k",) for err in errors)

    def test_top_k_above_maximum_raises_error(self):
        """Test that top_k > 100 raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", top_k=101)

        error = cast(ValidationError, exc_info.value)
        errors = error.errors()
        assert len(errors) > 0
        assert any(err["loc"] == ("top_k",) for err in errors)


class TestSearchRequestEdgeCases:
    """Test SearchRequest edge cases and combinations."""

    def test_very_long_query_is_valid(self):
        """Test that a very long query string is accepted."""
        long_query = "test " * 1000  # 5000 characters
        request = SearchRequest(query=long_query)

        assert request.query == long_query.strip()
        assert len(request.query) > 1000

    def test_query_with_special_characters(self):
        """Test that queries with special characters are valid."""
        special_query = "test @#$%^&*() query with 123 numbers!"
        request = SearchRequest(query=special_query)

        assert request.query == special_query

    def test_query_with_unicode_characters(self):
        """Test that queries with unicode characters are valid."""
        unicode_query = "test query with émojis 🔍 and ñoñó"
        request = SearchRequest(query=unicode_query)

        assert request.query == unicode_query

    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are caught together."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="", top_k=0)

        error = cast(ValidationError, exc_info.value)
        errors = error.errors()

        # Should have errors for query and top_k
        error_fields = {err["loc"][0] for err in errors}
        assert "query" in error_fields
        assert "top_k" in error_fields
