"""Document service for handling document upload and processing."""
import logging
from uuid import UUID, uuid4

from src.app.core.domain.models import Document
from src.app.core.services.document_processor import DocumentProcessor
from src.app.infrastructure.client_repository import ClientRepository
from src.app.infrastructure.document_repository import DocumentRepository
from src.shared.blob_storage.s3_blober import S3BlobStorage
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.exceptions import EntityNotFound

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for handling Document business logic."""

    def __init__(
        self,
        client_repository: ClientRepository,
        document_repository: DocumentRepository,
        unit_of_work: UnitOfWork,
        blob_storage: S3BlobStorage,
        document_processor: DocumentProcessor,
    ):
        """
        Initialize the document service.

        Args:
            client_repository: Repository for client operations
            document_repository: Repository for document operations
            unit_of_work: Unit of work for database transactions
            blob_storage: S3 blob storage for file operations
            document_processor: Processor for document chunking and embedding
        """
        self.client_repository = client_repository
        self.document_repository = document_repository
        self.unit_of_work = unit_of_work
        self.blob_storage = blob_storage
        self.document_processor = document_processor

    async def create_document(
        self,
        client_id: UUID,
        title: str,
        content: str
    ) -> Document:
        """
        Create a document record and upload it to S3.

        If the upload fails, the document is marked as FAILED. Otherwise, it's PENDING.

        Args:
            client_id: ID of the client who owns this document
            title: Document title
            content: Text content of the document

        Returns:
            The created Document domain model

        """
        client = await self.client_repository.get_by_id(client_id)
        if not client:
            raise EntityNotFound("Client", client_id)

        document_id = uuid4()
        s3_key = f"clients/{client_id}/documents/{document_id}.txt"
        document = Document(
            id=document_id,
            client_id=client_id,
            title=title,
            s3_key=s3_key,
        )

        try:
            await self.blob_storage.upload_text_content(s3_key, content)
        except RuntimeError as e:
            logger.error(f"S3 upload failed for document {document_id}: {e}")
            document.failed()

        async with self.unit_of_work:
            self.unit_of_work.add(document)

        return document

    async def get_document_by_id_and_client_id(
        self, document_id: UUID, client_id: UUID
    ) -> Document:
        """
        Get a document by its ID and client ID.

        Args:
            document_id: ID of the document to retrieve
            client_id: ID of the client owner

        Returns:
            The Document domain model

        Raises:
            ValueError: If document not found for the given client
        """
        document = await self.document_repository.get_client_document_by_id(
            document_id, client_id
        )
        if not document:
            raise EntityNotFound("Document", document_id)
        return document

    async def process_document(self, document_id: UUID, content: str) -> None:
        """
        Process a document by chunking its content, generating embeddings, and optionally summarizing.

        This method:
        1. Retrieves the document from the repository
        2. Uses DocumentProcessor to chunk, embed, and optionally summarize the content
        3. Updates document status to PROCESSED and sets the summary if available
        4. Persists all chunks in a single transaction

        This will be executed in a background process.

        Args:
            document_id: The ID of the document to process
            content: The raw text content to be chunked and embedded

        Raises:
            ValueError: If document not found or processing fails
        """
        logger.info("Starting processing for document %s", document_id)

        # Get the document
        document = await self.document_repository.get_by_id(document_id)
        if not document:
            logger.error("Document %s not found for processing", document_id)
            raise EntityNotFound("Document", document_id)

        # Process text to get chunks with embeddings and optional summary
        result = await self.document_processor.process_text(document_id, content)

        if result.summary:
            document.summarized(result.summary)

        document.processed()
        # Persist document status update and chunks in a single transaction
        async with self.unit_of_work:
            await self.unit_of_work.update(document)
            for chunk in result.chunks:
                self.unit_of_work.add(chunk)

        logger.info(
            "Successfully processed document %s with %d chunks%s",
            document_id,
            len(result.chunks),
            " and summary" if result.summary else ""
        )


    async def get_client_documents(self, client_id: UUID) -> list[Document]:
        """
        Get all documents for a specific client.

        Args:
            client_id: ID of the client

        Returns:
            List of Document domain models

        Raises:
            ValueError: If client not found
        """
        # Verify client exists
        client = await self.client_repository.get_by_id(client_id)
        if not client:
            raise EntityNotFound("Client", client_id)

        return await self.document_repository.get_by_client_id(client_id)
