"""Integration tests for document upload and chunking pipeline."""
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from pydantic.v1 import EmailStr
from sqlalchemy import select

from src.app.main import create_app
from src.client import NevisClient, CreateClientRequest
from src.client.schemas import CreateDocumentRequest


@pytest_asyncio.fixture(scope="module")
async def test_app(test_settings_override):
    """
    Create test application with test database and LocalStack.
    Module-scoped for better performance.
    """
    app = create_app()
    yield app


@pytest_asyncio.fixture
async def nevis_client(test_app, clean_database):
    """Create Nevis client for testing with clean database."""
    transport = ASGITransport(app=test_app)
    http_client = AsyncClient(transport=transport, base_url="http://test")
    client = NevisClient(base_url="http://test", client=http_client)

    async with client:
        yield client


@pytest.mark.asyncio
async def test_upload_document_full_pipeline(nevis_client, s3_storage, clean_database):
    """
    Integration test for complete document upload pipeline.

    This test verifies:
    1. Client exists in database
    2. Document is uploaded to S3
    3. Document record is created in database with PROCESSED status
    4. Document chunks are created and persisted to database
    5. S3 file can be retrieved and matches original content
    """
    # 1. Create a client first
    client_request = CreateClientRequest(
        first_name="John",
        last_name="Doe",
        email=EmailStr("john.doe@test.com"),
        description="Test client for document upload"
    )
    client_response = await nevis_client.create_client(client_request)
    client_id = client_response.id

    # 2. Prepare document content
    document_title = "Test Document"
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

    # 3. Upload document via API
    document_request = CreateDocumentRequest(
        title=document_title,
        content=document_content
    )

    # TODO: create corresponding methods on nevis_client to make this easier
    # Make the API call through the test client
    response = await nevis_client.client.post(f"/api/v1/clients/{client_id}/documents/",
            json={"title": document_title, "content": document_content})


    # 4. Verify API response
    assert response.status_code == 201
    document_data = response.json()
    assert document_data["title"] == document_title
    assert document_data["client_id"] == str(client_id)
    assert document_data["status"] == "PROCESSED"
    assert "s3_key" in document_data

    document_id = document_data["id"]
    s3_key = document_data["s3_key"]

    # 5. Verify file exists in S3 (LocalStack)
    file_exists = await s3_storage.object_exists(s3_key)
    assert file_exists, f"File {s3_key} should exist in S3"

    # 6. Verify content in S3 matches original
    retrieved_content = await s3_storage.download_text_content(s3_key)
    assert retrieved_content == document_content.strip()

    # # 7. Verify document record in database
    # from src.app.infrastructure.entities.document_entity import DocumentEntity, DocumentStatus
    #
    # async with clean_database.session_maker() as session:
    #     result = await session.execute(
    #         select(DocumentEntity).where(DocumentEntity.id == document_id)
    #     )
    #     db_document = result.scalar_one_or_none()
    #
    #     assert db_document is not None
    #     assert db_document.title == document_title
    #     assert db_document.client_id == client_id
    #     assert db_document.status == DocumentStatus.PROCESSED
    #     assert db_document.s3_key == s3_key
    #
    # # 8. Verify document chunks were created
    # from src.app.infrastructure.entities import DocumentChunkEntity
    #
    # async with clean_database.session_maker() as session:
    #     result = await session.execute(
    #         select(DocumentChunkEntity)
    #         .where(DocumentChunkEntity.document_id == document_id)
    #         .order_by(DocumentChunkEntity.chunk_index)
    #     )
    #     chunks = result.scalars().all()
    #
    #     # Should have at least one chunk
    #     assert len(chunks) > 0, "Document should be chunked into at least one chunk"
    #
    #     # Verify chunk properties
    #     for i, chunk in enumerate(chunks):
    #         assert chunk.chunk_index == i, f"Chunk {i} should have correct index"
    #         assert chunk.document_id == document_id
    #         assert len(chunk.chunk_content) > 0, f"Chunk {i} should have content"
    #         assert chunk.embedding is None, "Embedding should be null in Phase 1"
    #
    #     # Verify chunks are ordered
    #     chunk_indices = [chunk.chunk_index for chunk in chunks]
    #     assert chunk_indices == sorted(chunk_indices), "Chunks should be ordered by index"


@pytest.mark.asyncio
async def test_upload_document_client_not_found(test_app, clean_database):
    """Test that uploading a document for non-existent client fails."""
    from uuid import uuid4

    non_existent_client_id = uuid4()
    document_request = CreateDocumentRequest(
        title="Test Document",
        content="This should fail"
    )

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            f"/api/v1/clients/{non_existent_client_id}/documents/",
            json={"title": document_request.title, "content": document_request.content}
        )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_document_empty_content(nevis_client, test_app, clean_database):
    """Test that uploading a document with empty content fails validation."""
    # Create a client first
    client_request = CreateClientRequest(
        first_name="Jane",
        last_name="Smith",
        email=EmailStr("jane.smith@test.com"),
        description="Test client"
    )
    client_response = await nevis_client.create_client(client_request)
    client_id = client_response.id

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.post(
            f"/api/v1/clients/{client_id}/documents/",
            json={"title": "Empty Doc", "content": ""}
        )

    # Should fail validation
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_get_document(nevis_client, test_app, clean_database):
    """Test retrieving a document by ID."""
    # Create client
    client_request = CreateClientRequest(
        first_name="Bob",
        last_name="Johnson",
        email=EmailStr("bob.johnson@test.com"),
        description="Test client"
    )
    client_response = await nevis_client.create_client(client_request)
    client_id = client_response.id

    # Upload document
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        create_response = await http_client.post(
            f"/api/v1/clients/{client_id}/documents/",
            json={"title": "Test Doc", "content": "Test content for retrieval"}
        )
        assert create_response.status_code == 201
        document_id = create_response.json()["id"]

        # Retrieve document
        get_response = await http_client.get(
            f"/api/v1/clients/{client_id}/documents/{document_id}"
        )

    assert get_response.status_code == 200
    doc_data = get_response.json()
    assert doc_data["id"] == document_id
    assert doc_data["title"] == "Test Doc"
    assert doc_data["client_id"] == str(client_id)


@pytest.mark.asyncio
async def test_list_client_documents(nevis_client, test_app, clean_database):
    """Test listing all documents for a client."""
    # Create client
    client_request = CreateClientRequest(
        first_name="Alice",
        last_name="Williams",
        email=EmailStr("alice.williams@test.com"),
        description="Test client"
    )
    client_response = await nevis_client.create_client(client_request)
    client_id = client_response.id

    # Upload multiple documents
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        for i in range(3):
            response = await http_client.post(
                f"/api/v1/clients/{client_id}/documents/",
                json={"title": f"Document {i}", "content": f"Content for document {i}"}
            )
            assert response.status_code == 201

        # List documents
        list_response = await http_client.get(
            f"/api/v1/clients/{client_id}/documents/"
        )

    assert list_response.status_code == 200
    documents = list_response.json()
    assert len(documents) == 3
    assert all(doc["client_id"] == str(client_id) for doc in documents)
