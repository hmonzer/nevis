"""Integration tests for document upload and chunking pipeline."""
import pytest
from pydantic.v1 import EmailStr

from src.client import CreateClientRequest
from src.client.schemas import CreateDocumentRequest


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

    document_response = await nevis_client.upload_document(client_id, document_request)

    # 4. Verify API response
    assert document_response.title == document_title
    assert document_response.client_id == client_id
    assert document_response.status == "PENDING"  # It's pending because processing is in the background
    assert document_response.s3_key
    s3_key = document_response.s3_key

    # 5. Verify file exists in S3 (LocalStack)
    file_exists = await s3_storage.object_exists(s3_key)
    assert file_exists, f"File {s3_key} should exist in S3"

    # 6. Verify content in S3 matches original
    retrieved_content = await s3_storage.download_text_content(s3_key)
    assert retrieved_content == document_content.strip()

@pytest.mark.asyncio
async def test_upload_document_client_not_found(nevis_client):
    """Test that uploading a document for non-existent client fails."""
    from uuid import uuid4
    import httpx

    non_existent_client_id = uuid4()
    document_request = CreateDocumentRequest(
        title="Test Document", content="This should fail"
    )

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await nevis_client.upload_document(non_existent_client_id, document_request)

    assert isinstance(excinfo.value, httpx.HTTPStatusError)
    assert excinfo.value.response.status_code == 404
    assert "not found" in excinfo.value.response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_document(nevis_client):
    """Test retrieving a document by ID."""
    # Create client
    client_request = CreateClientRequest(
        first_name="Bob",
        last_name="Johnson",
        email=EmailStr("bob.johnson@test.com"),
        description="Test client",
    )
    client_response = await nevis_client.create_client(client_request)
    client_id = client_response.id

    # Upload document
    doc_request = CreateDocumentRequest(
        title="Test Doc", content="Test content for retrieval"
    )
    uploaded_doc = await nevis_client.upload_document(client_id, doc_request)
    document_id = uploaded_doc.id

    # Retrieve document
    retrieved_doc = await nevis_client.get_document(client_id, document_id)

    assert retrieved_doc.id == document_id
    assert retrieved_doc.title == "Test Doc"
    assert retrieved_doc.client_id == client_id


@pytest.mark.asyncio
async def test_list_client_documents(nevis_client):
    """Test listing all documents for a client."""
    # Create client
    client_request = CreateClientRequest(
        first_name="Alice",
        last_name="Williams",
        email=EmailStr("alice.williams@test.com"),
        description="Test client",
    )
    client_response = await nevis_client.create_client(client_request)
    client_id = client_response.id

    # Upload multiple documents
    for i in range(3):
        doc_request = CreateDocumentRequest(
            title=f"Document {i}", content=f"Content for document {i}"
        )
        await nevis_client.upload_document(client_id, doc_request)

    # List documents
    documents = await nevis_client.list_documents(client_id)

    assert len(documents) == 3
    assert all(doc.client_id == client_id for doc in documents)
