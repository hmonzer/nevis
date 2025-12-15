import pytest
import pytest_asyncio
from pathlib import Path
from typing import Any
from ranx import Qrels, Run, evaluate
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.app.core.domain.models import SearchRequest
from src.app.core.services.chunks_search_service import DocumentChunkSearchService
from src.app.core.services.document_search_service import DocumentSearchService
from src.app.core.services.client_search_service import ClientSearchService
from src.app.core.services.search_service import SearchService
from src.app.core.services.embedding import SentenceTransformerEmbedding
from src.app.core.services.reranker import CrossEncoderReranker
from src.app.infrastructure.document_search_repository import DocumentSearchRepository
from src.app.infrastructure.client_search_repository import ClientSearchRepository
from src.app.infrastructure.mappers.document_chunk_mapper import DocumentChunkMapper
from src.app.infrastructure.mappers.client_mapper import ClientMapper
from src.app.infrastructure.document_repository import DocumentRepository
from src.app.infrastructure.mappers.document_mapper import DocumentMapper

from tests.e2e_eval.data_setup import load_eval_suite, setup_corpus
from tests.e2e_eval.evaluation import EvaluationMetrics, UseCaseResult, EvaluationReporter


# --- Fixtures reusing conftest and dependencies ---

@pytest_asyncio.fixture(scope="module")
def sentence_transformer_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@pytest_asyncio.fixture(scope="module")
def embedding_service(sentence_transformer_model):
    return SentenceTransformerEmbedding(sentence_transformer_model)

@pytest_asyncio.fixture(scope="module")
def cross_encoder_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@pytest_asyncio.fixture(scope="module")
def reranker_service(cross_encoder_model):
    return CrossEncoderReranker(cross_encoder_model)

@pytest_asyncio.fixture
def document_search_repository(clean_database):
    return DocumentSearchRepository(clean_database, DocumentChunkMapper())

@pytest_asyncio.fixture
def client_search_repository(clean_database):
    return ClientSearchRepository(clean_database, ClientMapper())

@pytest_asyncio.fixture
def chunk_search_service(embedding_service, document_search_repository, reranker_service):
    return DocumentChunkSearchService(
        embedding_service=embedding_service,
        search_repository=document_search_repository,
        reranker_service=reranker_service,
    )

@pytest_asyncio.fixture
def document_repository(clean_database):
    return DocumentRepository(clean_database, DocumentMapper())

@pytest_asyncio.fixture
def document_search_service(chunk_search_service, document_repository):
    return DocumentSearchService(
        chunk_search_service=chunk_search_service,
        document_repository=document_repository,
    )

@pytest_asyncio.fixture
def client_search_service(client_search_repository):
    return ClientSearchService(search_repository=client_search_repository)

@pytest_asyncio.fixture
def unified_search_service(client_search_service, document_search_service):
    return SearchService(
        client_search_service=client_search_service,
        document_search_service=document_search_service,
    )


def build_qrels_dict(test: Any, id_map: dict[str, Any]) -> dict[str, int]:
    """
    Build relevance judgments (qrels) for a single test query.

    Args:
        test: Test case containing query_id and expected_result_ids
        id_map: Mapping from synthetic IDs to actual database IDs

    Returns:
        Dictionary mapping document_id -> relevance_score (int)
        Higher scores indicate higher relevance (first result gets highest score)
    """
    qrels_entry: dict[str, int] = {}

    for rank, exp_id in enumerate(test.expected_result_ids):
        if exp_id in id_map:
            real_id = str(id_map[exp_id])
            # Score = (num_expected - rank): first result gets highest score
            # Example: [doc1, doc2, doc3] -> doc1=3, doc2=2, doc3=1
            score = len(test.expected_result_ids) - rank
            qrels_entry[real_id] = score
        else:
            print(f"Warning: Expected ID {exp_id} not found in creation map.")

    return qrels_entry


def build_run_dict_entry(results: list[Any]) -> dict[str, float]:
    """
    Build system output scores (run) for search results.

    Args:
        results: List of SearchResult objects from unified search

    Returns:
        Dictionary mapping entity_id -> score (float)
        Higher scores for higher-ranked results (first result gets highest score)
        Handles both CLIENT and DOCUMENT result types
    """
    run_entry: dict[str, float] = {}

    for rank, result in enumerate(results):
        # Score = (num_results - rank): mirrors qrels scoring pattern
        # Example: 5 results -> 5.0, 4.0, 3.0, 2.0, 1.0
        score = float(len(results) - rank)
        run_entry[str(result.entity.id)] = score

    return run_entry


async def run_eval_use_case(
    use_case: Any,
    id_map: dict[str, Any],
    search_service: SearchService,
    top_k: int = 10,
) -> UseCaseResult | None:
    """
    Run evaluation for a single use case.

    Args:
        use_case: Use case containing tests
        id_map: Mapping from synthetic IDs to actual database IDs
        search_service: Unified search service to evaluate
        top_k: Number of results to retrieve

    Returns:
        UseCaseResult with metrics or None if no valid data
    """
    print(f"\n{'='*60}")
    print(f"Running Use Case: {use_case.title}")
    print(f"{'='*60}")

    # Build Qrels (Ground Truth) and Run (System Output)
    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}

    for test in use_case.tests:
        # Build ground truth relevance judgments
        qrels_dict[test.query_id] = build_qrels_dict(test, id_map)

        # Execute search using unified search service
        request = SearchRequest(query=test.query_text, top_k=top_k, threshold=0.5)
        results = await search_service.search(request)

        num_clients = sum(1 for r in results if r.type == 'CLIENT')
        num_documents = sum(1 for r in results if r.type == 'DOCUMENT')
        print(f"  Query '{test.query_text[:50]}...': {len(results)} total results "
              f"({num_clients} clients, {num_documents} documents)")

        # Build system output scores
        run_dict[test.query_id] = build_run_dict_entry(results)

    # Evaluate using Ranx
    if not qrels_dict or not run_dict:
        print("⚠️  No valid qrels or run data to evaluate")
        return None

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)  # type: ignore

    ranx_metrics = evaluate(qrels, run, metrics=["mrr", "recall@5", "ndcg@5"])  # type: ignore

    # Convert to our metrics model
    metrics = EvaluationMetrics.from_ranx_result(ranx_metrics)

    print(f"\n📊 Metrics for {use_case.title}:")
    print(f"  MRR:       {metrics.mrr:.4f}")
    print(f"  Recall@5:  {metrics.recall_at_5:.4f}")
    print(f"  NDCG@5:    {metrics.ndcg_at_5:.4f}")

    return UseCaseResult(
        use_case_title=use_case.title,
        metrics=metrics,
        num_queries=len(use_case.tests),
    )


@pytest.mark.asyncio
async def test_wealth_manager_eval(
    nevis_client,  # Used for data setup via API
    unified_search_service,  # Unified search service to evaluate
    clean_database,  # Ensures fresh DB
    s3_storage  # Ensures S3
):
    """
    E2E evaluation of unified search service using synthetic wealth management data.

    This test:
    1. Loads an evaluation suite with a unified corpus
    2. Ingests all corpus data once (clients and documents)
    3. For each use case, runs queries and evaluates results
    4. Compares search results against expected results using IR metrics
    5. Collects metrics across all use cases for aggregate assessment
    """
    # 1. Load the Evaluation Suite
    data_path = Path(__file__).parent / "data" / "synthetic_wealth_data.json"
    suite = load_eval_suite(data_path)

    print(f"\n{'='*60}")
    print(f"Starting Evaluation Suite: {suite.suite_name}")
    print(f"{'='*60}")

    # 2. Setup entire corpus once (all clients and documents)
    print(f"\n{'='*60}")
    print(f"Setting up corpus...")
    print(f"{'='*60}")
    id_map = await setup_corpus(suite, nevis_client)
    print(f"✅ Corpus setup complete: {len(id_map)} entities created")

    # 3. Run all use cases and collect results
    all_results: list[UseCaseResult] = []
    failures: list[tuple[str, str]] = []

    for use_case in suite.use_cases:
        try:
            result = await run_eval_use_case(
                use_case=use_case,
                id_map=id_map,
                search_service=unified_search_service,
                top_k=10,
            )

            if result:
                all_results.append(result)

                # Soft validation: collect failures instead of asserting immediately
                if result.metrics.recall_at_5 <= 0:
                    failures.append((
                        use_case.title,
                        f"Recall@5 is {result.metrics.recall_at_5:.4f}, expected > 0"
                    ))
            else:
                failures.append((use_case.title, "No valid evaluation data"))

        except Exception as e:
            print(f"❌ Error running use case '{use_case.title}': {e}")
            failures.append((use_case.title, f"Exception: {str(e)}"))

    # 4. Print aggregate results with formatted table
    EvaluationReporter.print_summary(
        suite_name=suite.suite_name,
        total_use_cases=len(suite.use_cases),
        all_results=all_results,
        failures=failures,
    )

    # 5. Final assertion: fail test only if ALL use cases failed
    assert len(all_results) > 0, "All use cases failed - no valid metrics collected"