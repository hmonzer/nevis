import pytest
from uuid import uuid4


from src.app.core.domain.models import Client, Document, DocumentStatus
from src.app.infrastructure.document_repository import DocumentRepository
from src.app.infrastructure.mappers.document_mapper import DocumentMapper


@pytest.fixture
def document_repository(clean_database) -> DocumentRepository:
    """Fixture for DocumentRepository."""
    return DocumentRepository(clean_database, DocumentMapper())


@pytest.mark.asyncio
async def test_get_document_by_id(document_repository, unit_of_work):
    """Test retrieving a document by its ID."""
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="Client",
        email="test.client@test.com",
    )
    doc = Document(
        id=uuid4(),
        client_id=client.id,
        title="Test Document",
        s3_key="test/key.txt",
        status=DocumentStatus.PENDING,
    )
    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(doc)

    retrieved_doc = await document_repository.get_by_id(doc.id)
    assert retrieved_doc is not None
    assert retrieved_doc.id == doc.id
    assert retrieved_doc.title == doc.title


@pytest.mark.asyncio
async def test_get_documents_by_client_id(document_repository, unit_of_work):
    """Test retrieving all documents for a given client."""
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="Client",
        email="test@test.com",
    )
    docs = [
        Document(
            id=uuid4(),
            client_id=client.id,
            title=f"Doc {i}",
            s3_key=f"test/key{i}.txt",
            status=DocumentStatus.PENDING,
        )
        for i in range(3)
    ]
    async with unit_of_work:
        unit_of_work.add(client)
        for doc in docs:
            unit_of_work.add(doc)

    retrieved_docs = await document_repository.get_by_client_id(client.id)
    assert len(retrieved_docs) == 3
    assert {doc.id for doc in retrieved_docs} == {doc.id for doc in docs}


@pytest.mark.asyncio
async def test_get_document_by_id_and_client_id(document_repository, unit_of_work):
    """Test retrieving a document by both ID and client ID."""
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="Client",
        email="test@test.com",
    )
    doc = Document(
        id=uuid4(),
        client_id=client.id,
        title="Specific Document",
        s3_key="specific/key.txt",
        status=DocumentStatus.PROCESSED,
    )
    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(doc)

    # Should find the document
    retrieved_doc = await document_repository.get_client_document_by_id(
        doc.id, client.id
    )
    assert retrieved_doc is not None
    assert retrieved_doc.id == doc.id

    # Should not find with wrong client ID
    retrieved_doc = await document_repository.get_client_document_by_id(
        doc.id, uuid4()
    )
    assert retrieved_doc is None


@pytest.mark.asyncio
async def test_get_documents_by_ids(document_repository, unit_of_work):
    """Test batch fetching multiple documents by their IDs."""
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="Client",
        email="test@test.com",
    )

    # Create 5 documents
    docs = [
        Document(
            id=uuid4(),
            client_id=client.id,
            title=f"Document {i}",
            s3_key=f"test/doc{i}.txt",
            status=DocumentStatus.PENDING,
        )
        for i in range(5)
    ]

    async with unit_of_work:
        unit_of_work.add(client)
        for doc in docs:
            unit_of_work.add(doc)

    # Test fetching all 5 documents
    all_ids = [doc.id for doc in docs]
    retrieved_docs = await document_repository.get_by_ids(all_ids)

    assert len(retrieved_docs) == 5
    retrieved_ids = {doc.id for doc in retrieved_docs}
    assert retrieved_ids == set(all_ids)

    # Verify titles are correct
    retrieved_titles = {doc.title for doc in retrieved_docs}
    expected_titles = {f"Document {i}" for i in range(5)}
    assert retrieved_titles == expected_titles


@pytest.mark.asyncio
async def test_get_documents_by_ids_partial_match(document_repository, unit_of_work):
    """Test batch fetching when some document IDs don't exist."""
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="Client",
        email="test@test.com",
    )

    # Create 3 documents
    docs = [
        Document(
            id=uuid4(),
            client_id=client.id,
            title=f"Document {i}",
            s3_key=f"test/doc{i}.txt",
            status=DocumentStatus.PENDING,
        )
        for i in range(3)
    ]

    async with unit_of_work:
        unit_of_work.add(client)
        for doc in docs:
            unit_of_work.add(doc)

    # Request 3 existing documents + 2 non-existent ones
    existing_ids = [doc.id for doc in docs]
    non_existent_ids = [uuid4(), uuid4()]
    mixed_ids = existing_ids + non_existent_ids

    retrieved_docs = await document_repository.get_by_ids(mixed_ids)

    # Should only return the 3 existing documents
    assert len(retrieved_docs) == 3
    retrieved_ids = {doc.id for doc in retrieved_docs}
    assert retrieved_ids == set(existing_ids)

    # Non-existent IDs should not cause errors
    for non_existent_id in non_existent_ids:
        assert non_existent_id not in retrieved_ids


@pytest.mark.asyncio
async def test_get_documents_by_ids_empty_list(document_repository):
    """Test batch fetching with empty list returns empty result."""
    retrieved_docs = await document_repository.get_by_ids([])
    assert len(retrieved_docs) == 0
    assert retrieved_docs == []


@pytest.mark.asyncio
async def test_get_documents_by_ids_single_document(document_repository, unit_of_work):
    """Test batch fetching with a single document ID."""
    client = Client(
        id=uuid4(),
        first_name="Test",
        last_name="Client",
        email="test@test.com",
    )

    doc = Document(
        id=uuid4(),
        client_id=client.id,
        title="Single Document",
        s3_key="test/single.txt",
        status=DocumentStatus.PROCESSED,
    )

    async with unit_of_work:
        unit_of_work.add(client)
        unit_of_work.add(doc)

    # Fetch single document using batch method
    retrieved_docs = await document_repository.get_by_ids([doc.id])

    assert len(retrieved_docs) == 1
    assert retrieved_docs[0].id == doc.id
    assert retrieved_docs[0].title == "Single Document"
    assert retrieved_docs[0].status == DocumentStatus.PROCESSED
