"""Document summarization service using LLM providers."""
import logging
from abc import ABC, abstractmethod

import anthropic
import google.generativeai as genai

logger = logging.getLogger(__name__)

SUMMARIZATION_PROMPT_TEMPLATE = """You are assisting a wealth manager who needs quick document summaries.
Summarize the following document in {max_words} words or fewer.

Focus on information relevant to wealth management:
- Financial details (amounts, accounts, investments)
- Client identity and contact information
- Legal or compliance-relevant content
- Key dates and deadlines
- Asset or liability information

If the document is not financial in nature, provide a general summary of the key points.

Document:
{content}

Summary:"""


class SummarizationService(ABC):
    """Abstract base class for document summarization."""

    @abstractmethod
    async def summarize(self, content: str) -> str:
        """
        Generate a summary of the document content.

        Args:
            content: The full document text to summarize.

        Returns:
            A summary of the document content.

        Raises:
            SummarizationError: If summarization fails.
        """
        pass


class SummarizationError(Exception):
    """Exception raised when summarization fails."""
    pass


class ClaudeSummarizationService(SummarizationService):
    """Summarization service using Anthropic's Claude API."""

    def __init__(self, api_key: str, model: str, max_words: int = 100, max_tokens: int = 200):
        """
        Initialize the Claude summarization service.

        Args:
            api_key: Anthropic API key.
            model: Claude model to use for summarization.
            max_words: Target maximum words for the summary.
            max_tokens: Maximum tokens for the LLM response.

        Raises:
            ValueError: If model is not provided.
        """
        if not model:
            raise ValueError("Claude model must be specified")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_words = max_words
        self.max_tokens = max_tokens

    async def summarize(self, content: str) -> str:
        """Generate a summary using Claude."""
        try:
            prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(content=content, max_words=self.max_words)
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = message.content[0].text.strip()
            logger.info("Generated summary with Claude (%d chars)", len(summary))
            return summary
        except anthropic.APIError as e:
            logger.error("Claude API error during summarization: %s", e)
            raise SummarizationError(f"Claude summarization failed: {e}") from e


class GeminiSummarizationService(SummarizationService):
    """Summarization service using Google's Gemini API."""

    def __init__(self, api_key: str, model: str, max_words: int = 100, max_tokens: int = 200):
        """
        Initialize the Gemini summarization service.

        Args:
            api_key: Google API key.
            model: Gemini model to use for summarization.
            max_words: Target maximum words for the summary.
            max_tokens: Maximum tokens for the LLM response (not used by Gemini but kept for consistency).

        Raises:
            ValueError: If model is not provided.
        """
        if not model:
            raise ValueError("Gemini model must be specified")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.max_words = max_words
        self.max_tokens = max_tokens

    async def summarize(self, content: str) -> str:
        """Generate a summary using Gemini."""
        try:
            prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(content=content, max_words=self.max_words)
            response = await self.model.generate_content_async(prompt)
            summary = response.text.strip()
            logger.info("Generated summary with Gemini (%d chars)", len(summary))
            return summary
        except Exception as e:
            logger.error("Gemini API error during summarization: %s", e)
            raise SummarizationError(f"Gemini summarization failed: {e}") from e