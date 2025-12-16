"""Document summarization service using LLM providers."""
import logging
from abc import ABC, abstractmethod

import anthropic
import google.generativeai as genai

logger = logging.getLogger(__name__)

MAX_SUMMARY_WORDS = 100
SUMMARIZATION_PROMPT = f"""You are assisting a wealth manager who needs quick document summaries.
Summarize the following document in {MAX_SUMMARY_WORDS} words or fewer.

Focus on information relevant to wealth management:
- Financial details (amounts, accounts, investments)
- Client identity and contact information
- Legal or compliance-relevant content
- Key dates and deadlines
- Asset or liability information

If the document is not financial in nature, provide a general summary of the key points.

Document:
{{content}}

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
            A summary of the document content (max 100 words).

        Raises:
            SummarizationError: If summarization fails.
        """
        pass


class SummarizationError(Exception):
    """Exception raised when summarization fails."""
    pass


class ClaudeSummarizationService(SummarizationService):
    """Summarization service using Anthropic's Claude API."""

    def __init__(self, api_key: str, model: str):
        """
        Initialize the Claude summarization service.

        Args:
            api_key: Anthropic API key.
            model: Claude model to use for summarization.

        Raises:
            ValueError: If model is not provided.
        """
        if not model:
            raise ValueError("Claude model must be specified")
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def summarize(self, content: str) -> str:
        """Generate a summary using Claude."""
        try:
            prompt = SUMMARIZATION_PROMPT.format(content=content)
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=200,
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

    def __init__(self, api_key: str, model: str):
        """
        Initialize the Gemini summarization service.

        Args:
            api_key: Google API key.
            model: Gemini model to use for summarization.

        Raises:
            ValueError: If model is not provided.
        """
        if not model:
            raise ValueError("Gemini model must be specified")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    async def summarize(self, content: str) -> str:
        """Generate a summary using Gemini."""
        try:
            prompt = SUMMARIZATION_PROMPT.format(content=content)
            response = await self.model.generate_content_async(prompt)
            summary = response.text.strip()
            logger.info("Generated summary with Gemini (%d chars)", len(summary))
            return summary
        except Exception as e:
            logger.error("Gemini API error during summarization: %s", e)
            raise SummarizationError(f"Gemini summarization failed: {e}") from e