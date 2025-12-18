"""Unit tests for DocumentService."""
import pytest
from uuid import uuid4

from pydantic.v1 import EmailStr

from src.app.core.domain.models import Client
from src.app.core.services.document_service import DocumentService
from src.shared.exceptions import EntityNotFound


@pytest.fixture
def document_service_instance(
    test_container,
    unit_of_work,
    s3_storage,
    client_repository,
    document_repository,
):
    """Create a DocumentService instance for testing."""
    return DocumentService(
        client_repository=client_repository,
        document_repository=document_repository,
        unit_of_work=unit_of_work,
        blob_storage=s3_storage,
        document_processor=test_container.document_processor(),
    )


@pytest.fixture
async def test_client_with_document(document_service_instance, unit_of_work):
    """Create a test client with a document for testing download."""
    # Create client
    client = Client(
        id=uuid4(),
        first_name="Download",
        last_name="Test",
        email="download.test@example.com",
        description="Test client for download functionality",
    )
    async with unit_of_work:
        unit_of_work.add(client)

    # Create document
    document_content = "This is test content for download URL generation."
    document = await document_service_instance.create_document(
        client_id=client.id,
        title="Download Test Document",
        content=document_content,
    )

    return client, document, document_content


@pytest.mark.asyncio
async def test_get_document_download_url(document_service_instance, test_client_with_document):
    """Test generating a pre-signed download URL for a document."""
    client, document, _ = test_client_with_document

    # Get download URL
    url = await document_service_instance.get_document_download_url(
        document_id=document.id,
        client_id=client.id,
    )

    # Verify URL is returned
    assert url is not None
    assert isinstance(url, str)
    assert document.s3_key in url


@pytest.mark.asyncio
async def test_get_document_download_url_custom_expiration(
    document_service_instance, test_client_with_document
):
    """Test generating a download URL with custom expiration."""
    client, document, _ = test_client_with_document

    # Get download URL with 5 minute expiration
    url = await document_service_instance.get_document_download_url(
        document_id=document.id,
        client_id=client.id,
        expiration=300,
    )

    assert url is not None
    assert isinstance(url, str)


@pytest.mark.asyncio
async def test_get_document_download_url_document_not_found(document_service_instance):
    """Test that EntityNotFound is raised for non-existent document."""
    non_existent_document_id = uuid4()
    non_existent_client_id = uuid4()

    with pytest.raises(EntityNotFound) as exc_info:
        await document_service_instance.get_document_download_url(
            document_id=non_existent_document_id,
            client_id=non_existent_client_id,
        )

    assert "Document" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_document_download_url_wrong_client(
    document_service_instance, test_client_with_document
):
    """Test that EntityNotFound is raised when document belongs to different client."""
    _, document, _ = test_client_with_document
    wrong_client_id = uuid4()

    with pytest.raises(EntityNotFound) as exc_info:
        await document_service_instance.get_document_download_url(
            document_id=document.id,
            client_id=wrong_client_id,
        )

    assert "Document" in str(exc_info.value)
