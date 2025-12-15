import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any
from uuid import UUID

from src.client.nevis_client import NevisClient
from src.client.schemas import CreateClientRequest, CreateDocumentRequest, DocumentStatusEnum
from tests.e2e_eval.schema import EvalSuite

logger = logging.getLogger(__name__)

def load_eval_suite(file_path: Path) -> EvalSuite:
    with open(file_path, "r") as f:
        data = json.load(f)
    return EvalSuite(**data)

async def wait_for_document_processing(
    client_api: NevisClient,
    client_id: UUID,
    document_id: UUID,
    timeout_sec: int = 10
) -> None:
    """Polls document status until processed or timeout."""
    for _ in range(timeout_sec * 5): # Check every 0.2s
        doc = await client_api.get_document(client_id, document_id)
        if doc.status == DocumentStatusEnum.PROCESSED:
            return
        if doc.status == DocumentStatusEnum.FAILED:
            raise RuntimeError(f"Document {document_id} processing failed.")
        await asyncio.sleep(0.2)
    raise RuntimeError(f"Document {document_id} processing timed out after {timeout_sec}s")

async def setup_corpus(suite: EvalSuite, client_api: NevisClient) -> Dict[str, UUID]:
    """
    Populates the database and S3 with the entire corpus using the API.
    This loads all clients and documents from the suite once.

    Returns a mapping of {json_id: real_system_id} for both clients and documents.
    """
    id_map: Dict[str, UUID] = {}

    # 1. Create all Clients
    logger.info(f"Creating {len(suite.corpus.clients)} clients...")
    for client_record in suite.corpus.clients:
        request = CreateClientRequest(
            first_name=client_record.first_name,
            last_name=client_record.last_name,
            email=client_record.email,
            description=client_record.description
        )

        response = await client_api.create_client(request)
        id_map[client_record.id] = response.id
        logger.info(f"Created client {response.email} with ID {response.id} (mapped from {client_record.id})")

    # 2. Create all Documents
    logger.info(f"Creating {len(suite.corpus.documents)} documents...")
    for doc_record in suite.corpus.documents:
        real_client_id = id_map.get(doc_record.client_id)
        if not real_client_id:
            logger.error(f"Client ID {doc_record.client_id} not found for document {doc_record.title}")
            continue

        request = CreateDocumentRequest(
            title=doc_record.title,
            content=doc_record.content
        )

        response = await client_api.upload_document(real_client_id, request)

        # Wait for processing to complete
        await wait_for_document_processing(client_api, real_client_id, response.id)

        id_map[doc_record.id] = response.id
        logger.info(f"Created and processed document '{response.title}' with ID {response.id} (mapped from {doc_record.id})")

    logger.info(f"Corpus setup complete. Created {len(id_map)} total entities.")
    return id_map
