"""Document service for handling document upload and processing."""
from uuid import UUID, uuid4
from datetime import datetime, UTC

from sqlalchemy.exc import IntegrityError

from src.app.core.domain.models import Document, DocumentChunk, DocumentStatus
from src.app.core.services.chunking import ChunkingStrategy
from src.shared.database.unit_of_work import UnitOfWork
from src.shared.blob_storage.s3_blober import S3BlobStorage
from src.app.infrastructure.client_repository import ClientRepository
from src.app.infrastructure.document_repository import DocumentRepository


# TODO: Add tests for Document Service that verify the document is persisted to S3, document entity is persisted with proper state.
class DocumentService:
    """Service for handling Document business logic."""

    def __init__(
        self,
        client_repository: ClientRepository,
        document_repository: DocumentRepository,
        unit_of_work: UnitOfWork,
        blob_storage: S3BlobStorage,
        chunking_strategy: ChunkingStrategy,
    ):
        """
        Initialize the document service.

        Args:
            client_repository: Repository for client operations
            document_repository: Repository for document operations
            unit_of_work: Unit of work for database transactions
            blob_storage: S3 blob storage for file operations
            chunking_strategy: Service for text chunking
        """
        self.client_repository = client_repository
        self.document_repository = document_repository
        self.unit_of_work = unit_of_work
        self.blob_storage = blob_storage
        self.chunking_service = chunking_strategy

    async def upload_document(
        self,
        client_id: UUID,
        title: str,
        content: str
    ) -> Document:
        """
        Upload a document, store it in S3, chunk it, and persist everything to the database.

        Process:
        1. Verify client exists
        2. Upload content to S3
        3. Create document record with PENDING status
        4. Chunk the content
        5. Persist document and chunks to database
        6. Update document status to PROCESSED

        Args:
            client_id: ID of the client who owns this document
            title: Document title
            content: Text content of the document

        Returns:
            The created Document domain model

        Raises:
            ValueError: If client doesn't exist or processing fails
        """
        # 1. Verify client exists
        client = await self.client_repository.get_by_id(client_id)
        if not client:
            raise ValueError(f"Client with ID {client_id} not found")

        # 2. Generate document ID and S3 key
        document_id = uuid4()
        s3_key = f"clients/{client_id}/documents/{document_id}.txt"

        try:
            # 3. Upload content to S3
            await self.blob_storage.upload_text_content(s3_key, content)
        except RuntimeError as e:
            raise ValueError(f"Failed to upload document to storage: {str(e)}") from e

        # 4. Create document model with PENDING status
        document = Document(
            id=document_id,
            client_id=client_id,
            title=title,
            s3_key=s3_key,
            status=DocumentStatus.PENDING,
            created_at=datetime.now(UTC),
        )

        # TODO: at this point, let's persist the document to the DB with Pending status and refactor the chunking into a separate method.
        #  Later this chunking will happen in a background thread. It will retrieve the document, do the chunking and calculate embeddings,
        #  then update the doc status to PROCESSED.
        try:
            # 5. Chunk the content
            chunk_texts = self.chunking_service.chunk_text(content)

            # 6. Create DocumentChunk models
            chunks: list[DocumentChunk] = []
            for index, chunk_text in enumerate(chunk_texts):
                chunk = DocumentChunk(
                    id=uuid4(),
                    document_id=document_id,
                    chunk_index=index,
                    chunk_content=chunk_text,
                    embedding=None,  # Phase 1: embeddings are null
                )
                chunks.append(chunk)

            # 7. Update document status to PROCESSED
            document.status = DocumentStatus.PROCESSED

            # 8. Persist document and chunks using unit of work
            async with self.unit_of_work:
                self.unit_of_work.add(document)
                for chunk in chunks:
                    self.unit_of_work.add(chunk)

        except IntegrityError as e:
            # If database persistence fails, try to clean up S3
            try:
                await self.blob_storage.delete_object(s3_key)
            except RuntimeError:
                pass  # Ignore cleanup errors

            raise ValueError(f"Failed to persist document: {str(e)}") from e
        except Exception as e:
            # If any other error occurs, mark as FAILED and try to persist
            document.status = DocumentStatus.FAILED
            try:
                async with self.unit_of_work:
                    self.unit_of_work.add(document)
            except Exception:
                pass  # Ignore errors when saving failed status

            raise ValueError(f"Failed to process document: {str(e)}") from e

        return document

    async def get_document(self, document_id: UUID) -> Document:
        """
        Get a document by ID.

        Args:
            document_id: ID of the document to retrieve

        Returns:
            The Document domain model

        Raises:
            ValueError: If document not found
        """
        document = await self.document_repository.get_by_id(document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} not found")
        return document

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
            raise ValueError(f"Client with ID {client_id} not found")

        return await self.document_repository.get_by_client_id(client_id)
