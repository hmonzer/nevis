import json
import logging
import asyncio
from pathlib import Path
from typing import Dict
from uuid import UUID

from src.app.core.domain.models import DocumentStatus
from src.app.infrastructure.document_repository import DocumentRepository
from src.client.nevis_client import NevisClient
from src.client.schemas import CreateClientRequest, CreateDocumentRequest
from tests.e2e_eval.schema import EvalSuite

logger = logging.getLogger(__name__)

def load_eval_suite(file_path: Path) -> EvalSuite:
    with open(file_path, "r") as f:
        data = json.load(f)
    return EvalSuite(**data)

async def wait_for_documents_processing(
    document_repository: DocumentRepository,
    document_ids: list[UUID],
    timeout_sec: int = 30
) -> None:
    """
    Wait for all documents to be processed using batch fetching.

    Uses the repository's batch fetch method for efficient status checking,
    reducing N API calls to a single database query per poll iteration.

    Args:
        document_repository: Repository for batch document fetching
        document_ids: List of document IDs to wait for
        timeout_sec: Maximum time to wait for all documents

    Raises:
        RuntimeError: If any document fails processing or timeout is reached
    """
    if not document_ids:
        return

    for attempt in range(timeout_sec * 5):  # Check every 0.2s
        # Batch fetch all documents in a single query
        documents = await document_repository.get_by_ids(document_ids)

        # Check if we got all documents
        if len(documents) != len(document_ids):
            missing_ids = set(document_ids) - {doc.id for doc in documents}
            logger.warning(f"Some documents not found: {missing_ids}")

        # Check status of all documents
        all_processed = True
        for doc in documents:
            if doc.status == DocumentStatus.FAILED:
                raise RuntimeError(f"Document {doc.id} processing failed.")
            if doc.status != DocumentStatus.PROCESSED:
                all_processed = False
                break

        if all_processed and len(documents) == len(document_ids):
            logger.info(f"✅ All {len(document_ids)} documents processed successfully")
            return

        await asyncio.sleep(0.2)

    raise RuntimeError(f"Document processing timed out after {timeout_sec}s")

async def setup_corpus(
    suite: EvalSuite,
    client_api: NevisClient,
    document_repository: DocumentRepository
) -> Dict[str, UUID]:
    """
    Populates the database and S3 with the entire corpus using the API.
    This loads all clients and documents from the suite once, using parallel async operations
    for maximum performance.

    Args:
        suite: Evaluation suite containing corpus data
        client_api: API client for creating clients and documents
        document_repository: Repository for efficient batch status checking

    Returns:
        A mapping of {json_id: real_system_id} for both clients and documents.
    """
    id_map: Dict[str, UUID] = {}

    # 1. Create all Clients IN PARALLEL
    logger.info(f"Creating {len(suite.corpus.clients)} clients in parallel...")

    client_tasks = []
    for client_record in suite.corpus.clients:
        request = CreateClientRequest(
            first_name=client_record.first_name,
            last_name=client_record.last_name,
            email=client_record.email,
            description=client_record.description
        )
        client_tasks.append(client_api.create_client(request))

    # Execute all client creations in parallel
    client_responses = await asyncio.gather(*client_tasks)

    # Map IDs
    for client_record, response in zip(suite.corpus.clients, client_responses):
        id_map[client_record.id] = response.id
        logger.info(f"Created client {response.email} with ID {response.id} (mapped from {client_record.id})")

    # 2. Create all Documents IN PARALLEL (without waiting for processing)
    logger.info(f"Creating {len(suite.corpus.documents)} documents in parallel...")

    document_tasks = []
    document_metadata = []  # Track (doc_record, real_client_id) for later mapping

    for doc_record in suite.corpus.documents:
        real_client_id = id_map.get(doc_record.client_id)
        if not real_client_id:
            logger.error(f"Client ID {doc_record.client_id} not found for document {doc_record.title}")
            continue

        request = CreateDocumentRequest(
            title=doc_record.title,
            content=doc_record.content
        )

        document_tasks.append(client_api.upload_document(real_client_id, request))
        document_metadata.append((doc_record, real_client_id))

    # Execute all document creations in parallel
    document_responses = await asyncio.gather(*document_tasks)

    # Map document IDs immediately (don't wait for processing yet)
    for (doc_record, real_client_id), response in zip(document_metadata, document_responses):
        id_map[doc_record.id] = response.id
        logger.info(f"Created document '{response.title}' with ID {response.id} (mapped from {doc_record.id})")

    # 3. Wait for ALL documents to be processed using batch repository fetch
    logger.info(f"Waiting for {len(document_responses)} documents to be processed...")

    document_ids = [resp.id for resp in document_responses]
    await wait_for_documents_processing(document_repository, document_ids, timeout_sec=30)

    logger.info(f"Corpus setup complete. Created {len(id_map)} total entities.")
    return id_map
