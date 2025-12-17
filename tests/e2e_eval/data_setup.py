import json
import logging
import asyncio
from pathlib import Path
from typing import Dict
from uuid import UUID

from src.client.nevis_client import NevisClient
from src.client.schemas import CreateClientRequest, CreateDocumentRequest, DocumentStatusEnum
from tests.e2e_eval.schema import EvalSuite

logger = logging.getLogger(__name__)

def load_eval_suite(file_path: Path) -> EvalSuite:
    with open(file_path, "r") as f:
        data = json.load(f)
    return EvalSuite(**data)

async def wait_for_documents_processing(
    nevis_client: NevisClient,
    client_ids: list[UUID],
    expected_doc_count: int,
    timeout_sec: int = 30
) -> None:
    """
    Wait for all documents to be processed by polling via the API.

    Uses the nevis_client to list documents for each client and check their status.
    This allows the evaluation to run against any remote Nevis API instance.

    Args:
        nevis_client: API client for fetching documents
        client_ids: List of client IDs whose documents to check
        expected_doc_count: Total number of documents expected across all clients
        timeout_sec: Maximum time to wait for all documents

    Raises:
        RuntimeError: If any document fails processing or timeout is reached
    """
    if not client_ids or expected_doc_count == 0:
        return

    poll_interval = 0.5  # seconds between polls
    max_attempts = int(timeout_sec / poll_interval)

    for attempt in range(max_attempts):
        all_documents = []

        # Fetch documents for all clients
        for client_id in client_ids:
            try:
                docs = await nevis_client.list_documents(client_id)
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Error fetching documents for client {client_id}: {e}")

        # Check if we have all expected documents
        if len(all_documents) < expected_doc_count:
            await asyncio.sleep(poll_interval)
            continue

        # Check status of all documents
        all_processed = True
        for doc in all_documents:
            if doc.status == DocumentStatusEnum.FAILED:
                raise RuntimeError(f"Document {doc.id} ({doc.title}) processing failed.")
            if doc.status != DocumentStatusEnum.PROCESSED:
                all_processed = False
                break

        if all_processed:
            logger.info(f"✅ All {len(all_documents)} documents processed successfully")
            return

        await asyncio.sleep(poll_interval)

    raise RuntimeError(f"Document processing timed out after {timeout_sec}s")

async def setup_corpus(
    suite: EvalSuite,
    nevis_client: NevisClient,
) -> Dict[str, UUID]:
    """
    Populates the database and S3 with the entire corpus using the API.
    This loads all clients and documents from the suite once, using parallel async operations
    for maximum performance.

    Args:
        suite: Evaluation suite containing corpus data
        nevis_client: API client for creating clients and documents

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
        client_tasks.append(nevis_client.create_client(request))

    # Execute all client creations in parallel
    client_responses = await asyncio.gather(*client_tasks)

    # Map IDs and collect client UUIDs for later polling
    created_client_ids: list[UUID] = []
    for client_record, response in zip(suite.corpus.clients, client_responses):
        id_map[client_record.id] = response.id
        created_client_ids.append(response.id)
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

        document_tasks.append(nevis_client.upload_document(real_client_id, request))
        document_metadata.append((doc_record, real_client_id))

    # Execute all document creations in parallel
    document_responses = await asyncio.gather(*document_tasks)

    # Map document IDs immediately (don't wait for processing yet)
    for (doc_record, real_client_id), response in zip(document_metadata, document_responses):
        id_map[doc_record.id] = response.id
        logger.info(f"Created document '{response.title}' with ID {response.id} (mapped from {doc_record.id})")

    # 3. Wait for ALL documents to be processed by polling via API
    logger.info(f"Waiting for {len(document_responses)} documents to be processed...")

    await wait_for_documents_processing(
        nevis_client=nevis_client,
        client_ids=created_client_ids,
        expected_doc_count=len(document_responses),
        timeout_sec=60  # Larger documents may need more time
    )

    logger.info(f"Corpus setup complete. Created {len(id_map)} total entities.")
    return id_map
