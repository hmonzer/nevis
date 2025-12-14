"""Integration tests for end-to-end document search functionality."""
import pytest
import pytest_asyncio
from pydantic.v1 import EmailStr
from sentence_transformers import SentenceTransformer
from uuid import uuid4

from src.app.core.services.document_processor import DocumentProcessor
from src.app.core.services.document_service import DocumentService
from src.app.core.services.chunking import RecursiveChunkingStrategy
from src.app.core.services.embedding import SentenceTransformerEmbedding
from src.app.core.services.document_search_service import DocumentChunkSearchService
from src.app.infrastructure.document_search_repository import DocumentSearchRepository
from src.app.infrastructure.client_repository import ClientRepository
from src.app.infrastructure.document_repository import DocumentRepository
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper
from src.app.infrastructure.mappers.client_mapper import ClientMapper
from src.app.infrastructure.mappers.document_mapper import DocumentMapper
from src.shared.database.entity_mapper import EntityMapper
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.blob_storage.s3_blober import S3BlobStorage, S3BlobStorageSettings
from src.app.core.domain.models import Client, Document, DocumentChunk


@pytest_asyncio.fixture(scope="module")
def sentence_transformer_model():
    """Create a SentenceTransformer model instance."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest_asyncio.fixture(scope="module")
def embedding_service(sentence_transformer_model):
    """Create an embedding service."""
    return SentenceTransformerEmbedding(sentence_transformer_model)


@pytest_asyncio.fixture(scope="module")
def chunking_service():
    """Create a chunking service."""
    return RecursiveChunkingStrategy(chunk_size=200, chunk_overlap=50)


@pytest_asyncio.fixture(scope="module")
def document_processor(chunking_service, embedding_service):
    """Create a document processor."""
    return DocumentProcessor(
        chunking_strategy=chunking_service,
        embedding_service=embedding_service,
    )


@pytest_asyncio.fixture
def search_repository(clean_database):
    """Create a document search repository."""
    return DocumentSearchRepository(clean_database, DocumentChunkMapper())


@pytest_asyncio.fixture
def search_service(embedding_service, search_repository):
    """Create a document chunk search service."""
    return DocumentChunkSearchService(
        embedding_service=embedding_service,
        search_repository=search_repository
    )


@pytest_asyncio.fixture
async def entity_mapper():
    """Create entity mapper for unit of work."""
    return EntityMapper(
        entity_mappings={
            Client: ClientMapper().to_entity,
            Document: DocumentMapper().to_entity,
            DocumentChunk: DocumentChunkMapper().to_entity,
        }
    )


@pytest_asyncio.fixture
def client_repository(clean_database):
    """Create a client repository."""
    return ClientRepository(clean_database, ClientMapper())


@pytest_asyncio.fixture
def document_repository(clean_database):
    """Create a document repository."""
    return DocumentRepository(clean_database, DocumentMapper())


@pytest_asyncio.fixture
def unit_of_work_fixture(clean_database, entity_mapper):
    """Create a unit of work instance."""
    return UnitOfWork(clean_database, entity_mapper)


@pytest_asyncio.fixture
def s3_storage(s3_endpoint_url):
    """Create S3 storage instance for LocalStack."""
    settings = S3BlobStorageSettings(
        bucket_name="test-documents",
        endpoint_url=s3_endpoint_url,
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    return S3BlobStorage(settings)


@pytest_asyncio.fixture
def document_service(
    client_repository,
    document_repository,
    unit_of_work_fixture,
    s3_storage,
    document_processor
):
    """Create a document service."""
    return DocumentService(
        client_repository=client_repository,
        document_repository=document_repository,
        unit_of_work=unit_of_work_fixture,
        blob_storage=s3_storage,
        document_processor=document_processor
    )


@pytest.mark.asyncio
async def test_end_to_end_document_search_proof_of_address(
    clean_database,
    document_service,
    search_service,
    unit_of_work_fixture,
    localstack_container
):
    """
    Integration test for document search with proof of address use case.

    This test verifies:
    1. Multiple documents are uploaded and processed (utility bill, passport, driving license)
    2. Semantic search for "proof of address" returns relevant documents
    3. Utility bill ranks highest as it's the best proof of address
    """
    # 1. Create a client
    client = Client(
        id=uuid4(),
        first_name="John",
        last_name="Doe",
        email=EmailStr("john.doe@test.com"),
        description="Test client for proof of address search"
    )

    async with unit_of_work_fixture:
        unit_of_work_fixture.add(client)

    # 2. Create utility bill document (best proof of address)
    utility_bill_content = """
    ELECTRICITY BILL

    Account Holder: John Doe
    Service Address: 123 Main Street, Apartment 4B, Springfield, IL 62701
    Billing Period: October 1, 2024 - October 31, 2024

    This bill serves as proof of residence at the above address.
    The account holder is responsible for electricity usage at this residential address.

    Current Charges:
    - Energy Usage: $45.20
    - Distribution Charge: $12.50
    - Total Amount Due: $57.70

    Payment Due Date: November 15, 2024

    This utility bill can be used as proof of address for official purposes.
    Please retain this document for your records.
    """

    utility_bill = await document_service.create_document(
        client_id=client.id,
        title="Electricity Bill - October 2024",
        content=utility_bill_content
    )
    await document_service.process_document(utility_bill.id, utility_bill_content)

    # 3. Create passport document (proof of identity, NOT proof of address)
    passport_content = """
    PASSPORT

    Type: P (Regular Passport)
    Country Code: USA
    Passport No.: 123456789

    Surname: DOE
    Given Names: JOHN MICHAEL
    Nationality: UNITED STATES OF AMERICA
    Date of Birth: 15 MAR 1985
    Place of Birth: Chicago, Illinois
    Sex: M

    Date of Issue: 01 JAN 2020
    Date of Expiry: 01 JAN 2030
    Authority: U.S. Department of State

    This document is the property of the United States Government.
    It serves as proof of identity and citizenship.
    This passport is valid for international travel.
    """

    passport = await document_service.create_document(
        client_id=client.id,
        title="US Passport",
        content=passport_content
    )
    await document_service.process_document(passport.id, passport_content)

    # 4. Create driving license document (has address, but weaker proof than utility bill)
    license_content = """
    DRIVER LICENSE

    State of Illinois
    License Number: D123-4567-8901

    Name: JOHN MICHAEL DOE
    Address: 123 Main Street, Apt 4B
             Springfield, IL 62701

    Date of Birth: 03/15/1985
    Sex: M
    Height: 5'10"
    Eyes: Brown

    Class: D (Passenger Vehicles)
    Issue Date: 06/01/2023
    Expiration Date: 03/15/2029

    This license authorizes the holder to operate motor vehicles.
    Restrictions: Must wear corrective lenses
    """

    license = await document_service.create_document(
        client_id=client.id,
        title="Illinois Driver License",
        content=license_content
    )
    await document_service.process_document(license.id, license_content)

    # 5. Search for "proof of address"
    results = await search_service.search("proof of address", top_k=5)

    # 6. Verify results
    assert len(results) > 0, "Should find at least one relevant chunk"

    # Verify scores are in descending order
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score, "Results should be ordered by score descending"

    # Verify all scores are valid
    for result in results:
        assert 0.5 <= result.score <= 1.0, f"Score {result.score} should be between 0.5 (default) and 1.0"

    # 7. Verify utility bill chunks rank highest
    # Get the top result's document_id
    top_result_doc_id = results[0].chunk.document_id

    # The top result should be from the utility bill (which has explicit "proof of address" language)
    assert top_result_doc_id == utility_bill.id, (
        f"Top result should be from utility bill (best proof of address), "
        f"but got document {top_result_doc_id}"
    )


@pytest.mark.asyncio
async def test_search_with_similarity_threshold(
    clean_database,
    document_service,
    search_service,
    unit_of_work_fixture,
    localstack_container
):
    """
    Test that similarity threshold filters out low-relevance results.
    """
    # 1. Create a client
    client = Client(
        id=uuid4(),
        first_name="Jane",
        last_name="Smith",
        email=EmailStr("jane.smith@test.com"),
        description="Test client for threshold testing"
    )

    async with unit_of_work_fixture:
        unit_of_work_fixture.add(client)

    # 2. Create document with specific content
    document_content = """
    Bank Statement

    Account Holder: Jane Smith
    Account Number: 9876543210
    Statement Period: September 2024

    This bank statement shows the address on file for the account holder.
    Service Address: 456 Oak Avenue, Unit 12, Portland, OR 97201

    Opening Balance: $1,250.00
    Deposits: $2,500.00
    Withdrawals: $1,800.00
    Closing Balance: $1,950.00

    This document can serve as proof of residence and financial standing.
    """

    document = await document_service.create_document(
        client_id=client.id,
        title="Bank Statement - September 2024",
        content=document_content
    )
    await document_service.process_document(document.id, document_content)

    # 3. Search with high similarity threshold (0.7)
    results_high_threshold = await search_service.search(
        "proof of address",
        top_k=10,
        similarity_threshold=0.7  # High threshold
    )

    # 4. Search with low similarity threshold (0.3)
    results_low_threshold = await search_service.search(
        "proof of address",
        top_k=10,
        similarity_threshold=0.3  # Low threshold
    )

    # Assert - High threshold should return fewer or equal results
    assert len(results_high_threshold) <= len(results_low_threshold), (
        f"High threshold returned {len(results_high_threshold)} results, "
        f"low threshold returned {len(results_low_threshold)} results"
    )

    # All high threshold results should have score >= 0.7
    for result in results_high_threshold:
        assert result.score >= 0.7, f"Result with score {result.score} should be >= 0.7"


@pytest.mark.asyncio
async def test_search_empty_query_raises_error(search_service):
    """Test that searching with an empty query raises ValueError."""
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        await search_service.search("", top_k=10)

    with pytest.raises(ValueError, match="Search query cannot be empty"):
        await search_service.search("   ", top_k=10)


@pytest.mark.asyncio
async def test_search_invalid_top_k_raises_error(search_service):
    """Test that searching with invalid top_k raises ValueError."""
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        await search_service.search("test query", top_k=0)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        await search_service.search("test query", top_k=-1)


@pytest.mark.asyncio
async def test_search_invalid_threshold_raises_error(search_service):
    """Test that searching with invalid threshold raises ValueError."""
    with pytest.raises(ValueError, match="similarity_threshold must be between -1.0 and 1.0"):
        await search_service.search("test query", top_k=10, similarity_threshold=1.5)

    with pytest.raises(ValueError, match="similarity_threshold must be between -1.0 and 1.0"):
        await search_service.search("test query", top_k=10, similarity_threshold=-1.5)
