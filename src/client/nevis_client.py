"""Nevis HTTP Client for consuming the Nevis API."""
from uuid import UUID
from typing import Optional
from httpx import AsyncClient, Response

from src.client.schemas import (
    CreateClientRequest,
    ClientResponse,
    CreateDocumentRequest,
    DocumentResponse,
    DocumentDownloadResponse,
    SearchResultResponse,
)


class NevisClient:
    """HTTP client for interacting with the Nevis API."""

    def __init__(self, base_url: str, client: Optional[AsyncClient] = None):
        """
        Initialize the Nevis client.

        Args:
            base_url: Base URL of the Nevis API (e.g., "http://localhost:8000")
            client: Optional httpx.AsyncClient instance. If not provided, a new one will be created.
        """
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self):
        """Async context manager entry."""
        if self._owns_client:
            self._client = AsyncClient(base_url=self.base_url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._owns_client and self._client:
            await self._client.aclose()

    @property
    def client(self) -> AsyncClient:
        """Get the underlying httpx client."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async context manager.")
        return self._client

    async def create_client(self, request: CreateClientRequest) -> ClientResponse:
        """
        Create a new client.

        Args:
            request: Client creation request

        Returns:
            Created client response

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        response: Response = await self.client.post(
            "/api/v1/clients/",
            json=request.model_dump(mode="json")
        )
        response.raise_for_status()
        return ClientResponse(**response.json())

    async def get_client(self, client_id: UUID) -> ClientResponse:
        """
        Get a client by ID.

        Args:
            client_id: UUID of the client

        Returns:
            Client response

        Raises:
            httpx.HTTPStatusError: If the request fails (e.g., 404 if not found)
        """
        response: Response = await self.client.get(f"/api/v1/clients/{client_id}")
        response.raise_for_status()
        return ClientResponse(**response.json())

    async def upload_document(
        self, client_id: UUID, request: CreateDocumentRequest
    ) -> DocumentResponse:
        """
        Upload a document for a client.

        Args:
            client_id: UUID of the client
            request: Document creation request

        Returns:
            Created document response

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        response: Response = await self.client.post(
            f"/api/v1/clients/{client_id}/documents/",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()
        return DocumentResponse(**response.json())

    async def get_document(
        self, client_id: UUID, document_id: UUID
    ) -> DocumentResponse:
        """
        Get a document by ID for a client.

        Args:
            client_id: UUID of the client
            document_id: UUID of the document

        Returns:
            Document response

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        response: Response = await self.client.get(
            f"/api/v1/clients/{client_id}/documents/{document_id}"
        )
        response.raise_for_status()
        return DocumentResponse(**response.json())

    async def list_documents(self, client_id: UUID) -> list[DocumentResponse]:
        """
        List all documents for a client.

        Args:
            client_id: UUID of the client

        Returns:
            List of document responses

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        response: Response = await self.client.get(
            f"/api/v1/clients/{client_id}/documents/"
        )
        response.raise_for_status()
        return [DocumentResponse(**doc) for doc in response.json()]

    async def get_document_download_url(
        self, client_id: UUID, document_id: UUID
    ) -> DocumentDownloadResponse:
        """
        Get a pre-signed URL for downloading document content.

        Args:
            client_id: UUID of the client
            document_id: UUID of the document

        Returns:
            DocumentDownloadResponse with pre-signed S3 URL

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        response: Response = await self.client.get(
            f"/api/v1/clients/{client_id}/documents/{document_id}/download"
        )
        response.raise_for_status()
        return DocumentDownloadResponse(**response.json())

    async def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResultResponse]:
        """
        Search across clients and documents.

        Args:
            query: Search query string
            top_k: Maximum number of results to return (default: 10)

        Returns:
            List of search results containing matched clients and documents

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        response: Response = await self.client.get(
            "/api/v1/search/",
            params={"q": query, "top_k": top_k},
        )
        response.raise_for_status()
        return [SearchResultResponse(**result) for result in response.json()]

    async def get_documents(self, document_ids: list[UUID]) -> list[DocumentResponse]:
        """
        Fetch all documents by their IDs.
        Args:
             document_ids: document IDs
        Returns:
            list of documents
        Raises:
            httpx.HTTPStatusError: If the request fails
        """

        response = await self.client.get("/api/v1/documents", params={"document_ids": [str(id) for id in document_ids]})
        response.raise_for_status()
        return  [DocumentResponse(**doc) for doc in response.json()]