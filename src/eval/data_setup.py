"""Corpus setup for evaluation - handles data ingestion via API."""

import asyncio
import json
import logging
from pathlib import Path
from uuid import UUID

from src.client.nevis_client import NevisClient
from src.client.schemas import (
    CreateClientRequest,
    CreateDocumentRequest,
    DocumentStatusEnum,
)
from src.eval.schema import EvalSuite, Corpus, ClientRecord, DocumentRecord

logger = logging.getLogger(__name__)


def load_eval_suite(file_path: Path) -> EvalSuite:
    """Load evaluation suite from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return EvalSuite(**data)


class CorpusSetup:
    """
    Handles corpus setup for evaluation.

    Responsible for:
    - Creating clients via API
    - Creating documents via API
    - Waiting for document processing
    - Populating nevis_id on corpus records
    """

    def __init__(
        self,
        nevis_client: NevisClient,
        processing_timeout_sec: int = 60,
        poll_interval_sec: float = 0.5,
    ):
        """
        Initialize corpus setup.

        Args:
            nevis_client: API client for creating entities
            processing_timeout_sec: Max time to wait for document processing
            poll_interval_sec: Interval between status polls
        """
        self.nevis_client = nevis_client
        self.processing_timeout_sec = processing_timeout_sec
        self.poll_interval_sec = poll_interval_sec

    async def setup(self, suite: EvalSuite) -> Corpus:
        """
        Set up the entire corpus from an evaluation suite.

        Populates nevis_id on all ClientRecord and DocumentRecord objects.

        Args:
            suite: Evaluation suite containing corpus data

        Returns:
            The corpus with nevis_ids populated on all records
        """
        corpus = suite.corpus

        # Create clients and populate nevis_ids
        client_nevis_ids = await self._create_clients(corpus.clients)

        # Create documents and populate nevis_ids
        await self._create_documents(corpus.documents, corpus)

        # Wait for processing
        await self._wait_for_processing(
            client_ids=client_nevis_ids,
            expected_doc_count=len(corpus.documents),
        )

        logger.info(f"Corpus setup complete. Created {corpus.entity_count} entities.")
        return corpus

    async def _create_clients(self, clients: list[ClientRecord]) -> list[UUID]:
        """Create all clients in parallel and populate their nevis_ids."""
        logger.info(f"Creating {len(clients)} clients in parallel...")

        tasks = [
            self.nevis_client.create_client(
                CreateClientRequest(
                    first_name=client.first_name,
                    last_name=client.last_name,
                    email=client.email,
                    description=client.description,
                )
            )
            for client in clients
        ]

        responses = await asyncio.gather(*tasks)

        client_nevis_ids: list[UUID] = []
        for client, response in zip(clients, responses):
            client.nevis_id = response.id
            client_nevis_ids.append(response.id)
            logger.info(
                f"Created client {response.email} with nevis_id {response.id} "
                f"(input_id: {client.input_id})"
            )

        return client_nevis_ids

    async def _create_documents(
        self, documents: list[DocumentRecord], corpus: Corpus
    ) -> None:
        """Create all documents in parallel and populate their nevis_ids."""
        logger.info(f"Creating {len(documents)} documents in parallel...")

        tasks = []
        doc_records = []

        for doc in documents:
            # Look up the client's nevis_id from the corpus
            client = corpus.get_client_by_input_id(doc.client_input_id)
            if not client or not client.nevis_id:
                logger.error(
                    f"Client input_id {doc.client_input_id} not found or has no nevis_id "
                    f"for document {doc.title}"
                )
                continue

            tasks.append(
                self.nevis_client.upload_document(
                    client.nevis_id,
                    CreateDocumentRequest(title=doc.title, content=doc.content),
                )
            )
            doc_records.append(doc)

        responses = await asyncio.gather(*tasks)

        for doc, response in zip(doc_records, responses):
            doc.nevis_id = response.id
            logger.info(
                f"Created document '{response.title}' with nevis_id {response.id} "
                f"(input_id: {doc.input_id})"
            )

    async def _wait_for_processing(
        self,
        client_ids: list[UUID],
        expected_doc_count: int,
    ) -> None:
        """Wait for all documents to be processed."""
        if not client_ids or expected_doc_count == 0:
            return

        logger.info(f"Waiting for {expected_doc_count} documents to be processed...")

        max_attempts = int(self.processing_timeout_sec / self.poll_interval_sec)

        for attempt in range(max_attempts):
            all_documents = []

            for client_id in client_ids:
                try:
                    docs = await self.nevis_client.list_documents(client_id)
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Error fetching documents for client {client_id}: {e}")

            if len(all_documents) < expected_doc_count:
                await asyncio.sleep(self.poll_interval_sec)
                continue

            all_processed = True
            for doc in all_documents:
                if doc.status == DocumentStatusEnum.FAILED:
                    raise RuntimeError(
                        f"Document {doc.id} ({doc.title}) processing failed."
                    )
                if doc.status != DocumentStatusEnum.PROCESSED:
                    all_processed = False
                    break

            if all_processed:
                logger.info(
                    f"✅ All {len(all_documents)} documents processed successfully"
                )
                return

            await asyncio.sleep(self.poll_interval_sec)

        raise RuntimeError(
            f"Document processing timed out after {self.processing_timeout_sec}s"
        )
