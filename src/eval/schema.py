"""Data models for evaluation suites."""

from dataclasses import dataclass, field
from typing import Any, Union
from uuid import UUID

from pydantic import BaseModel, Field

from src.client.schemas import CreateClientRequest, CreateDocumentRequest


class ClientRecord(CreateClientRequest):
    """Client record with input ID and optional Nevis-generated ID."""

    input_id: str = Field(..., alias="id")
    nevis_id: UUID | None = None

    model_config = {"populate_by_name": True}


class DocumentRecord(CreateDocumentRequest):
    """Document record with input ID, client reference, and optional Nevis-generated ID."""

    input_id: str = Field(..., alias="id")
    client_input_id: str = Field(..., alias="client_id")
    nevis_id: UUID | None = None

    model_config = {"populate_by_name": True}


# Type alias for records that can be looked up
CorpusRecord = Union[ClientRecord, DocumentRecord]


class Corpus(BaseModel):
    """Evaluation corpus containing clients and documents with lookup capabilities."""

    clients: list[ClientRecord]
    documents: list[DocumentRecord]

    def get_client_by_input_id(self, input_id: str) -> ClientRecord | None:
        """Find a client by its input ID."""
        for client in self.clients:
            if client.input_id == input_id:
                return client
        return None

    def get_document_by_input_id(self, input_id: str) -> DocumentRecord | None:
        """Find a document by its input ID."""
        for doc in self.documents:
            if doc.input_id == input_id:
                return doc
        return None

    def get_record_by_input_id(self, input_id: str) -> CorpusRecord | None:
        """Find any record (client or document) by its input ID."""
        return self.get_client_by_input_id(input_id) or self.get_document_by_input_id(input_id)

    def get_record_by_nevis_id(self, nevis_id: UUID) -> CorpusRecord | None:
        """Find any record by its Nevis-generated ID."""
        for client in self.clients:
            if client.nevis_id == nevis_id:
                return client
        for doc in self.documents:
            if doc.nevis_id == nevis_id:
                return doc
        return None

    def get_input_id_by_nevis_id(self, nevis_id: UUID) -> str | None:
        """Get the input ID for a given Nevis ID."""
        record = self.get_record_by_nevis_id(nevis_id)
        if record:
            return record.input_id
        return None

    @property
    def all_records(self) -> list[CorpusRecord]:
        """Get all records (clients and documents)."""
        return list(self.clients) + list(self.documents)

    @property
    def entity_count(self) -> int:
        """Total number of entities in the corpus."""
        return len(self.clients) + len(self.documents)


@dataclass
class RetrievedResult:
    """A single retrieved result with its metadata."""

    input_id: str
    score: float
    result_type: str  # "CLIENT" or "DOCUMENT"

    def format_with_score(self) -> str:
        """Format as 'id(score)'."""
        return f"{self.input_id}({self.score:.3f})"

    def format_with_type_and_score(self) -> str:
        """Format as 'id(type:score)'."""
        return f"{self.input_id}({self.result_type}:{self.score:.3f})"


@dataclass
class TestResult:
    """Results from running a test query."""

    total_results: int
    num_clients: int
    num_documents: int
    retrieved: list[RetrievedResult] = field(default_factory=list)
    run_entry: dict[str, float] = field(default_factory=dict)

    @property
    def retrieved_ids(self) -> list[str]:
        """Get list of retrieved input IDs in order."""
        return [r.input_id for r in self.retrieved]


class TestItem(BaseModel):
    """Single test query with expected results and optional execution results."""

    query_id: str
    query_text: str
    expected_result_ids: list[str]

    # Populated after test execution
    result: TestResult | None = Field(default=None, exclude=True)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_negative_test(self) -> bool:
        """Whether this is a negative test expecting 0 results."""
        return len(self.expected_result_ids) == 0

    @property
    def has_result(self) -> bool:
        """Whether this test has been executed."""
        return self.result is not None

    def record_results(
        self,
        raw_results: list[Any],
        corpus: "Corpus",
    ) -> None:
        """
        Record search results from API response.

        Args:
            raw_results: List of SearchResult objects from unified search
            corpus: Corpus with populated nevis_ids for lookup
        """
        retrieved = []
        run_entry: dict[str, float] = {}

        for rank, result in enumerate(raw_results):
            nevis_id = result.entity.id
            input_id = corpus.get_input_id_by_nevis_id(nevis_id)
            if input_id is None:
                input_id = f"UNKNOWN_{str(nevis_id)[:8]}"

            retrieved.append(RetrievedResult(
                input_id=input_id,
                score=result.score,
                result_type=result.type,
            ))

            # Build run entry for ranx evaluation
            score = float(len(raw_results) - rank)
            run_entry[str(nevis_id)] = score

        num_clients = sum(1 for r in raw_results if r.type == "CLIENT")
        num_documents = sum(1 for r in raw_results if r.type == "DOCUMENT")

        self.result = TestResult(
            total_results=len(raw_results),
            num_clients=num_clients,
            num_documents=num_documents,
            retrieved=retrieved,
            run_entry=run_entry,
        )

    def build_qrels(self, corpus: "Corpus") -> dict[str, int]:
        """
        Build relevance judgments (qrels) for this test query.

        Args:
            corpus: Corpus with populated nevis_ids for lookup

        Returns:
            Dictionary mapping nevis_id -> relevance_score (int)
            Higher scores indicate higher relevance (first expected gets highest)
        """
        qrels_entry: dict[str, int] = {}

        for rank, input_id in enumerate(self.expected_result_ids):
            record = corpus.get_record_by_input_id(input_id)
            if record and record.nevis_id:
                nevis_id = str(record.nevis_id)
                # Score = (num_expected - rank): first result gets highest score
                score = len(self.expected_result_ids) - rank
                qrels_entry[nevis_id] = score

        return qrels_entry

    def get_comparison(self, top_k: int = 5) -> "TestComparison":
        """
        Compare expected vs retrieved results.

        Args:
            top_k: Number of top results to consider

        Returns:
            TestComparison with correct, missing, and extra IDs
        """
        if not self.has_result or self.result is None:
            return TestComparison(
                expected=set(self.expected_result_ids),
                retrieved_top_k=[],
                correct=set(),
                missing=set(self.expected_result_ids),
                extra=set(),
            )

        expected_set = set(self.expected_result_ids)
        retrieved_ids = self.result.retrieved_ids
        retrieved_top_k = retrieved_ids[:top_k]
        retrieved_set = set(retrieved_top_k)
        all_retrieved_set = set(retrieved_ids)

        return TestComparison(
            expected=expected_set,
            retrieved_top_k=retrieved_top_k,
            correct=expected_set & retrieved_set,
            missing=expected_set - all_retrieved_set,
            extra=retrieved_set - expected_set,
        )

    def has_issues(self, top_k: int = 5) -> bool:
        """Whether there are missing or extra results in top_k."""
        if not self.has_result:
            return True
        comparison = self.get_comparison(top_k)
        return bool(comparison.missing or comparison.extra)

    def get_extra_results_details(self, top_k: int = 5) -> list[str]:
        """
        Get formatted details for extra (unexpected) results.

        Returns:
            List of formatted strings like 'id(TYPE:score)'
        """
        if not self.has_result or self.result is None:
            return []

        comparison = self.get_comparison(top_k)
        details = []

        for r in self.result.retrieved[:top_k]:
            if r.input_id in comparison.extra:
                details.append(r.format_with_type_and_score())

        return details


@dataclass
class TestComparison:
    """Comparison between expected and retrieved results."""

    expected: set[str]
    retrieved_top_k: list[str]
    correct: set[str]
    missing: set[str]
    extra: set[str]

    @property
    def precision_str(self) -> str:
        """Format as 'correct/expected'."""
        return f"{len(self.correct)}/{len(self.expected)}"


class UseCase(BaseModel):
    """Use case grouping related test queries."""

    id: str
    title: str
    description: str | None = None
    tests: list[TestItem]


class EvalSuite(BaseModel):
    """Complete evaluation suite with corpus and test cases."""

    suite_name: str
    version: str
    description: str | None = None
    corpus: Corpus
    use_cases: list[UseCase]
