"""Unit tests for summarization service with mocked LLM calls."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.app.core.services.summarization import (
    ClaudeSummarizationService,
    GeminiSummarizationService,
    SummarizationError,
    SUMMARIZATION_PROMPT,
)


class TestClaudeSummarizationService:
    """Tests for Claude summarization service."""

    @pytest.mark.asyncio
    async def test_summarize_success(self):
        """Test successful summarization with Claude."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="This is a summary of the document.")]
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            service = ClaudeSummarizationService(
                api_key="test-api-key",
                model="claude-3-haiku-20240307"
            )
            service.client = mock_client

            result = await service.summarize("This is a long document content.")

            assert result == "This is a summary of the document."
            mock_client.messages.create.assert_called_once()
            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["model"] == "claude-3-haiku-20240307"
            assert call_kwargs["max_tokens"] == 200
            assert len(call_kwargs["messages"]) == 1
            assert "This is a long document content." in call_kwargs["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_summarize_strips_whitespace(self):
        """Test that summarization strips whitespace from response."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="  Summary with whitespace  \n")]
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            service = ClaudeSummarizationService(
                api_key="test-api-key",
                model="claude-3-haiku-20240307"
            )
            service.client = mock_client

            result = await service.summarize("Document content")

            assert result == "Summary with whitespace"

    @pytest.mark.asyncio
    async def test_summarize_api_error(self):
        """Test that API errors are wrapped in SummarizationError."""
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=anthropic.APIError(
                message="API Error",
                request=MagicMock(),
                body=None
            )
        )

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            service = ClaudeSummarizationService(
                api_key="test-api-key",
                model="claude-3-haiku-20240307"
            )
            service.client = mock_client

            with pytest.raises(SummarizationError) as exc_info:
                await service.summarize("Document content")

            assert "Claude summarization failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_custom_model(self):
        """Test using a custom Claude model."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Summary")]
        mock_client.messages.create = AsyncMock(return_value=mock_message)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            service = ClaudeSummarizationService(
                api_key="test-api-key",
                model="claude-3-sonnet-20240229"
            )
            service.client = mock_client

            await service.summarize("Content")

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["model"] == "claude-3-sonnet-20240229"

    def test_raises_error_when_no_model_provided(self):
        """Test that ValueError is raised when model is not provided."""
        with pytest.raises(ValueError) as exc_info:
            ClaudeSummarizationService(api_key="test-api-key", model="")

        assert "Claude model must be specified" in str(exc_info.value)


class TestGeminiSummarizationService:
    """Tests for Gemini summarization service."""

    @pytest.mark.asyncio
    async def test_summarize_success(self):
        """Test successful summarization with Gemini."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a Gemini summary."
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                service = GeminiSummarizationService(
                    api_key="test-api-key",
                    model="gemini-1.5-flash"
                )

                result = await service.summarize("Document content for Gemini")

                assert result == "This is a Gemini summary."
                mock_model.generate_content_async.assert_called_once()
                call_args = mock_model.generate_content_async.call_args[0][0]
                assert "Document content for Gemini" in call_args

    @pytest.mark.asyncio
    async def test_summarize_strips_whitespace(self):
        """Test that summarization strips whitespace from response."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "  Summary with whitespace  \n"
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                service = GeminiSummarizationService(
                    api_key="test-api-key",
                    model="gemini-1.5-flash"
                )

                result = await service.summarize("Document content")

                assert result == "Summary with whitespace"

    @pytest.mark.asyncio
    async def test_summarize_api_error(self):
        """Test that API errors are wrapped in SummarizationError."""
        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(
            side_effect=Exception("Gemini API Error")
        )

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel", return_value=mock_model):
                service = GeminiSummarizationService(
                    api_key="test-api-key",
                    model="gemini-1.5-flash"
                )

                with pytest.raises(SummarizationError) as exc_info:
                    await service.summarize("Document content")

                assert "Gemini summarization failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_custom_model(self):
        """Test using a custom Gemini model."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Summary"
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)

        with patch("google.generativeai.configure") as mock_configure:
            with patch("google.generativeai.GenerativeModel", return_value=mock_model) as mock_gen_model:
                service = GeminiSummarizationService(
                    api_key="test-api-key",
                    model="gemini-1.5-pro"
                )

                await service.summarize("Content")

                mock_configure.assert_called_once_with(api_key="test-api-key")
                mock_gen_model.assert_called_once_with("gemini-1.5-pro")

    def test_raises_error_when_no_model_provided(self):
        """Test that ValueError is raised when model is not provided."""
        with pytest.raises(ValueError) as exc_info:
            GeminiSummarizationService(api_key="test-api-key", model="")

        assert "Gemini model must be specified" in str(exc_info.value)


class TestSummarizationPrompt:
    """Tests for the summarization prompt template."""

    def test_prompt_contains_max_words(self):
        """Test that prompt mentions the max word limit."""
        assert "100" in SUMMARIZATION_PROMPT

    def test_prompt_has_content_placeholder(self):
        """Test that prompt has a content placeholder."""
        assert "{content}" in SUMMARIZATION_PROMPT

    def test_prompt_format_works(self):
        """Test that prompt can be formatted with content."""
        content = "Test document content"
        formatted = SUMMARIZATION_PROMPT.format(content=content)

        assert content in formatted
        assert "{content}" not in formatted
