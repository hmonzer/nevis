import logging

import pytest
from pathlib import Path
from typing import Any
from ranx import Qrels, Run, evaluate

from src.app.core.domain.models import SearchRequest
from src.app.core.services.search_service import SearchService

from tests.e2e_eval.data_setup import load_eval_suite, setup_corpus
from tests.e2e_eval.evaluation import EvaluationMetrics, UseCaseResult, EvaluationReporter


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


def log_query_results_compact(
    query_text: str,
    expected_ids: list[str],
    results: list[Any],
    reverse_map: dict[str, str],
    top_k: int = 5,
) -> None:
    """
    Log compact comparison of expected vs retrieved results.

    Args:
        query_text: The search query
        expected_ids: List of expected synthetic IDs
        results: List of SearchResult objects
        reverse_map: Mapping from UUID to synthetic ID
        top_k: Number of top results to consider for precision (default 5)
    """
    # Build retrieved synthetic IDs with scores for debugging
    retrieved_with_info = []
    for result in results:
        entity_uuid = str(result.entity.id)
        synthetic_id = reverse_map.get(entity_uuid, f"UNKNOWN_{entity_uuid[:8]}")
        retrieved_with_info.append((synthetic_id, result.score, result.type))

    retrieved_ids = [r[0] for r in retrieved_with_info]

    # Show comparison
    expected_set = set(expected_ids)
    retrieved_set = set(retrieved_ids[:top_k])  # Only consider top_k for precision
    all_retrieved_set = set(retrieved_ids)

    missing = expected_set - all_retrieved_set  # Expected but not retrieved at all
    extra = retrieved_set - expected_set  # Retrieved in top_k but not expected (hurts precision)
    correct = expected_set & retrieved_set

    # Only log if there are issues (missing OR extra docs)
    if missing or extra:
        # Format retrieved IDs with scores
        retrieved_with_scores = [f"{syn_id}({score:.3f})" for syn_id, score, _ in retrieved_with_info[:top_k]]

        print(f"    ⚠️  Query: '{query_text[:60]}'")
        print(f"        Expected: {', '.join(expected_ids)}")
        print(f"        Retrieved (top {top_k}): {', '.join(retrieved_with_scores)}")
        print(f"        ✅ Correct: {len(correct)}/{len(expected_set)}", end="")

        if missing:
            print(f" | ❌ Missing: {', '.join(missing)}", end="")

        if extra:
            # Show extra docs with their scores and types for debugging
            extra_details = []
            for syn_id, score, result_type in retrieved_with_info[:top_k]:
                if syn_id in extra:
                    extra_details.append(f"{syn_id}({result_type}:{score:.3f})")
            print(f" | 🔴 Extra (hurts precision): {', '.join(extra_details)}", end="")

        print()  # Newline at the end


async def run_eval_use_case(
    use_case: Any,
    id_map: dict[str, Any],
    reverse_map: dict[str, str],
    search_service: SearchService,
    top_k: int = 10,
) -> UseCaseResult | None:
    """
    Run evaluation for a single use case.

    Args:
        use_case: Use case containing tests
        id_map: Mapping from synthetic IDs to actual database IDs
        reverse_map: Mapping from UUIDs to synthetic IDs (for logging)
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
        request = SearchRequest(query=test.query_text, top_k=top_k, threshold=0.3)
        results = await search_service.search(request)

        num_clients = sum(1 for r in results if r.type == 'CLIENT')
        num_documents = sum(1 for r in results if r.type == 'DOCUMENT')
        print(f"  Query '{test.query_text[:50]}...': {len(results)} total results "
              f"({num_clients} clients, {num_documents} documents)")

        # Log details for queries with issues (missing or extra docs)
        log_query_results_compact(test.query_text, test.expected_result_ids, results, reverse_map, top_k)

        # Build system output scores
        run_dict[test.query_id] = build_run_dict_entry(results)

    # Evaluate using Ranx
    if not qrels_dict or not run_dict:
        print("⚠️  No valid qrels or run data to evaluate")
        return None

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)  # type: ignore

    ranx_metrics = evaluate(qrels, run, metrics=["mrr", "recall@5", "ndcg@5", "precision"])  # type: ignore

    # Convert to our metrics model
    metrics = EvaluationMetrics.from_ranx_result(ranx_metrics)

    print(f"\n📊 Metrics for {use_case.title}:")
    print(f"  MRR:       {metrics.mrr:.4f}")
    print(f"  Recall@5:  {metrics.recall_at_5:.4f}")
    print(f"  NDCG@5:    {metrics.ndcg_at_5:.4f}")
    print(f"  Precision:    {metrics.precision:.4f}")

    return UseCaseResult(
        use_case_title=use_case.title,
        metrics=metrics,
        num_queries=len(use_case.tests),
    )


@pytest.mark.asyncio
async def test_wealth_manager_eval(caplog,
    nevis_client,  # Used for data setup via API
    unified_search_service,  # Unified search service to evaluate
    document_repository,  # Repository for efficient batch document status checking
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
    caplog.set_level(logging.CRITICAL)
    # 1. Load the Evaluation Suite
    data_path = Path(__file__).parent / "data" / "synthetic_wealth_data.json"
    suite = load_eval_suite(data_path)

    print(f"\n{'='*60}")
    print(f"Starting Evaluation Suite: {suite.suite_name}")
    print(f"{'='*60}")

    # 2. Setup entire corpus once (all clients and documents)
    print(f"\n{'='*60}")
    print("Setting up corpus...")
    print(f"{'='*60}")
    id_map = await setup_corpus(suite, nevis_client, document_repository)
    # Create reverse map for logging (UUID -> synthetic ID)
    reverse_map = {str(uuid_val): synthetic_id for synthetic_id, uuid_val in id_map.items()}
    print(f"✅ Corpus setup complete: {len(id_map)} entities created")

    # 3. Run all use cases and collect results
    all_results: list[UseCaseResult] = []
    failures: list[tuple[str, str]] = []

    for use_case in suite.use_cases:
        try:
            result = await run_eval_use_case(
                use_case=use_case,
                id_map=id_map,
                reverse_map=reverse_map,
                search_service=unified_search_service,
                top_k=5,
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