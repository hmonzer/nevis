"""Integration tests for document upload and chunking pipeline."""
import pytest
from uuid import uuid4
from pydantic.v1 import EmailStr
from sentence_transformers import SentenceTransformer

from src.client import CreateClientRequest
from src.client.schemas import CreateDocumentRequest
from src.app.core.domain.models import Client
from src.app.core.services.document_processor import DocumentProcessor, ProcessingResult
from src.app.core.services.summarization import SummarizationService
from src.app.core.services.chunking import RecursiveChunkingStrategy
from src.app.core.services.embedding import SentenceTransformerEmbedding


# =============================================================================
# Shared Test Fixtures for Document Processing
# =============================================================================

@pytest.fixture(scope="module")
def embedding_model():
    """Shared embedding model (expensive to load)."""
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


@pytest.fixture
def chunking_service():
    """Chunking service with default settings."""
    return RecursiveChunkingStrategy(chunk_size=300, chunk_overlap=50)


@pytest.fixture
def embedding_service(embedding_model):
    """Embedding service using shared model."""
    return SentenceTransformerEmbedding(embedding_model)


@pytest.fixture
def document_processor(chunking_service, embedding_service):
    """Document processor without summarization."""
    return DocumentProcessor(
        chunking_strategy=chunking_service,
        embedding_service=embedding_service,
        summarization_service=None,
    )


@pytest.fixture
def document_processor_with_summary(chunking_service, embedding_service, mock_summarization_service):
    """Document processor with mock summarization."""
    return DocumentProcessor(
        chunking_strategy=chunking_service,
        embedding_service=embedding_service,
        summarization_service=mock_summarization_service,
    )


# =============================================================================
# API Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_upload_document_full_pipeline(nevis_client, s3_storage, clean_database):
    """
    Test complete document upload pipeline via API.

    Verifies: client creation, S3 upload, document creation, S3 content retrieval.
    """
    # Create client
    client_request = CreateClientRequest(
        first_name="John",
        last_name="Doe",
        email=EmailStr("john.doe@test.com"),
        description="Test client for document upload"
    )
    client_response = await nevis_client.create_client(client_request)
    client_id = client_response.id

    # Upload document
    document_content = """
    This is a test document for the document upload pipeline.
    It contains multiple paragraphs that should be chunked appropriately.

    The chunking strategy should split this content into meaningful chunks
    based on the configured chunk size and overlap settings.

    This test verifies that the entire pipeline works correctly:
    - S3 upload
    - Database persistence
    - Text chunking
    - Status updates
    """
    document_request = CreateDocumentRequest(title="Test Document", content=document_content)
    document_response = await nevis_client.upload_document(client_id, document_request)

    # Verify response
    assert document_response.title == "Test Document"
    assert document_response.client_id == client_id
    assert document_response.status == "PENDING"
    assert document_response.s3_key

    # Verify S3 content
    assert await s3_storage.object_exists(document_response.s3_key)
    retrieved_content = await s3_storage.download_text_content(document_response.s3_key)
    assert retrieved_content == document_content.strip()

@pytest.mark.asyncio
async def test_upload_document_client_not_found(nevis_client):
    """Test that uploading a document for non-existent client returns 404."""
    from typing import cast
    import httpx

    non_existent_client_id = uuid4()
    document_request = CreateDocumentRequest(title="Test Document", content="This should fail")

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await nevis_client.upload_document(non_existent_client_id, document_request)

    error = cast(httpx.HTTPStatusError, excinfo.value)
    assert error.response.status_code == 404
    assert "not found" in error.response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_document(nevis_client):
    """Test retrieving a document by ID."""
    # Create client and upload document
    client_request = CreateClientRequest(
        first_name="Bob", last_name="Johnson",
        email=EmailStr("bob.johnson@test.com"), description="Test client"
    )
    client_response = await nevis_client.create_client(client_request)

    doc_request = CreateDocumentRequest(title="Test Doc", content="Test content for retrieval")
    uploaded_doc = await nevis_client.upload_document(client_response.id, doc_request)

    # Retrieve and verify
    retrieved_doc = await nevis_client.get_document(client_response.id, uploaded_doc.id)
    assert retrieved_doc.id == uploaded_doc.id
    assert retrieved_doc.title == "Test Doc"
    assert retrieved_doc.client_id == client_response.id


@pytest.mark.asyncio
async def test_list_client_documents(nevis_client):
    """Test listing all documents for a client."""
    # Create client
    client_request = CreateClientRequest(
        first_name="Alice", last_name="Williams",
        email=EmailStr("alice.williams@test.com"), description="Test client"
    )
    client_response = await nevis_client.create_client(client_request)

    # Upload multiple documents
    for i in range(3):
        doc_request = CreateDocumentRequest(title=f"Document {i}", content=f"Content for document {i}")
        await nevis_client.upload_document(client_response.id, doc_request)

    # List and verify
    documents = await nevis_client.list_documents(client_response.id)
    assert len(documents) == 3
    assert all(doc.client_id == client_response.id for doc in documents)


# =============================================================================
# Summarization Test Fixtures
# =============================================================================

class MockSummarizationService(SummarizationService):
    """Mock summarization service for testing."""

    def __init__(self, summary_text: str = "This is a mock summary."):
        self.summary_text = summary_text
        self.call_count = 0

    async def summarize(self, content: str) -> str:
        self.call_count += 1
        return self.summary_text


@pytest.fixture
def mock_summarization_service():
    """Mock summarization service with wealth management summary."""
    return MockSummarizationService(
        summary_text="This document covers financial planning topics relevant to wealth management."
    )


# =============================================================================
# Document Processing Integration Tests
# =============================================================================

@pytest.mark.asyncio
async def test_document_processing_with_summary(document_processor_with_summary, mock_summarization_service):
    """Test that document processor generates summary when summarization service is provided."""
    content = """
    This is a test document for summarization.
    It contains multiple paragraphs that should be summarized.
    The summarization service should generate a concise summary.
    """

    result = await document_processor_with_summary.process_text(uuid4(), content)

    assert isinstance(result, ProcessingResult)
    assert result.summary == mock_summarization_service.summary_text
    assert mock_summarization_service.call_count == 1
    assert len(result.chunks) > 0


@pytest.mark.asyncio
async def test_document_processing_without_summary(document_processor):
    """Test that document processor works without summarization service."""
    content = "This is a test document without summarization."

    result = await document_processor.process_text(uuid4(), content)

    assert isinstance(result, ProcessingResult)
    assert result.summary is None
    assert len(result.chunks) > 0


@pytest.mark.asyncio
async def test_document_service_stores_summary(
    clean_database,
    unit_of_work,
    s3_storage,
    chunking_service,
    embedding_service,
):
    """Test that DocumentService stores summary in database after processing."""
    from src.app.core.services.document_service import DocumentService
    from src.app.infrastructure.client_repository import ClientRepository
    from src.app.infrastructure.document_repository import DocumentRepository
    from src.app.infrastructure.mappers.client_mapper import ClientMapper
    from src.app.infrastructure.mappers.document_mapper import DocumentMapper

    # Setup repositories
    client_repo = ClientRepository(clean_database, ClientMapper())
    document_repo = DocumentRepository(clean_database, DocumentMapper())

    # Create processor with mock summarization
    mock_summary = MockSummarizationService(
        summary_text="Summary: A test document about financial planning for retirement."
    )
    processor = DocumentProcessor(
        chunking_strategy=chunking_service,
        embedding_service=embedding_service,
        summarization_service=mock_summary,
    )

    document_service = DocumentService(
        client_repository=client_repo,
        document_repository=document_repo,
        unit_of_work=unit_of_work,
        blob_storage=s3_storage,
        document_processor=processor,
    )

    # Create client
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="User",
        email="test.user@example.com",
        description="Test client for summary integration",
    )
    async with unit_of_work:
        unit_of_work.add(client)

    # Create and process document
    content = """
    Financial Planning for Retirement
    This document outlines key strategies for retirement planning.
    """
    document = await document_service.create_document(
        client_id=client.id,
        title="Retirement Planning Guide",
        content=content,
    )
    await document_service.process_document(document.id, content)

    # Verify summary was stored
    retrieved_document = await document_repo.get_by_id(document.id)
    assert retrieved_document is not None
    assert retrieved_document.summary == mock_summary.summary_text
    assert retrieved_document.status.value == "PROCESSED"


@pytest.mark.asyncio
async def test_document_response_includes_summary_field(nevis_client):
    """Test that document API response includes summary field in schema."""
    # Create client and upload document
    client_request = CreateClientRequest(
        first_name="Summary", last_name="Test",
        email=EmailStr("summary.test@test.com"), description="Test client"
    )
    client_response = await nevis_client.create_client(client_request)

    doc_request = CreateDocumentRequest(
        title="Document With Summary Test",
        content="This is a document that will be processed and should have a summary field."
    )
    uploaded_doc = await nevis_client.upload_document(client_response.id, doc_request)

    # Retrieve and verify summary field exists in response schema
    retrieved_doc = await nevis_client.get_document(client_response.id, uploaded_doc.id)
    assert hasattr(retrieved_doc, 'summary') or 'summary' in retrieved_doc.model_fields
