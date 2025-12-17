"""
Reusable evaluation runner for search quality assessment.

This module provides the core evaluation logic that can be used by:
- pytest tests (via test fixtures)
- CLI scripts (against remote/docker-compose instances)
"""
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from ranx import Qrels, Run, evaluate

from src.client import NevisClient
from tests.e2e_eval.data_setup import load_eval_suite, setup_corpus
from tests.e2e_eval.evaluation import EvaluationMetrics, UseCaseResult, EvaluationReporter
from tests.e2e_eval.schema import EvalSuite


@dataclass
class EvaluationConfig:
    """Configuration for running an evaluation."""
    top_k: int = 5
    verbose: bool = True


@dataclass
class EvaluationResult:
    """Complete result of an evaluation run."""
    suite_name: str
    total_use_cases: int
    results: list[UseCaseResult]
    failures: list[tuple[str, str]]

    @property
    def success_rate(self) -> float:
        """Percentage of use cases that succeeded."""
        if self.total_use_cases == 0:
            return 0.0
        return len(self.results) / self.total_use_cases

    @property
    def average_metrics(self) -> dict[str, float] | None:
        """Calculate average metrics across all successful use cases."""
        if not self.results:
            return None
        return {
            "mrr": sum(r.metrics.mrr for r in self.results) / len(self.results),
            "recall_at_5": sum(r.metrics.recall_at_5 for r in self.results) / len(self.results),
            "ndcg_at_5": sum(r.metrics.ndcg_at_5 for r in self.results) / len(self.results),
            "precision": sum(r.metrics.precision for r in self.results) / len(self.results),
        }


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
    """
    run_entry: dict[str, float] = {}

    for rank, result in enumerate(results):
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
        top_k: Number of top results to consider for precision
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
    retrieved_set = set(retrieved_ids[:top_k])
    all_retrieved_set = set(retrieved_ids)

    missing = expected_set - all_retrieved_set
    extra = retrieved_set - expected_set
    correct = expected_set & retrieved_set

    # Only log if there are issues
    if missing or extra:
        retrieved_with_scores = [f"{syn_id}({score:.3f})" for syn_id, score, _ in retrieved_with_info[:top_k]]

        print(f"    ⚠️  Query: '{query_text[:60]}'")
        print(f"        Expected: {', '.join(expected_ids)}")
        print(f"        Retrieved (top {top_k}): {', '.join(retrieved_with_scores)}")
        print(f"        ✅ Correct: {len(correct)}/{len(expected_set)}", end="")

        if missing:
            print(f" | ❌ Missing: {', '.join(missing)}", end="")

        if extra:
            extra_details = []
            for syn_id, score, result_type in retrieved_with_info[:top_k]:
                if syn_id in extra:
                    extra_details.append(f"{syn_id}({result_type}:{score:.3f})")
            print(f" | 🔴 Extra (hurts precision): {', '.join(extra_details)}", end="")

        print()


async def run_eval_use_case(
    use_case: Any,
    id_map: dict[str, Any],
    reverse_map: dict[str, str],
    nevis_client: NevisClient,
    top_k: int = 10,
) -> UseCaseResult | None:
    """
    Run evaluation for a single use case.

    Args:
        use_case: Use case containing tests
        id_map: Mapping from synthetic IDs to actual database IDs
        reverse_map: Mapping from UUIDs to synthetic IDs (for logging)
        nevis_client: API client for search requests
        top_k: Number of results to retrieve

    Returns:
        UseCaseResult with metrics or None if no valid data
    """
    print(f"\n{'='*60}")
    print(f"Running Use Case: {use_case.title}")
    print(f"{'='*60}")

    qrels_dict: dict[str, dict[str, int]] = {}
    run_dict: dict[str, dict[str, float]] = {}

    negative_tests_passed = 0
    negative_tests_failed = 0

    for test in use_case.tests:
        results = await nevis_client.search(query=test.query_text, top_k=top_k)

        num_clients = sum(1 for r in results if r.type == 'CLIENT')
        num_documents = sum(1 for r in results if r.type == 'DOCUMENT')
        print(f"  Query '{test.query_text[:50]}...': {len(results)} total results "
              f"({num_clients} clients, {num_documents} documents)")

        # Handle negative test cases (queries expecting 0 results)
        if not test.expected_result_ids:
            if len(results) == 0:
                negative_tests_passed += 1
                print(f"    ✅ Negative test passed: correctly returned 0 results")
                continue
            else:
                negative_tests_failed += 1
                retrieved_ids = [reverse_map.get(str(r.entity.id), "UNKNOWN") for r in results[:5]]
                print(f"    ❌ Negative test failed: expected 0 results, got {len(results)}: {retrieved_ids}")
                qrels_dict[test.query_id] = {"__nonexistent_doc__": 1}
                run_dict[test.query_id] = build_run_dict_entry(results)
                continue

        # Normal case: build ground truth relevance judgments
        qrels_dict[test.query_id] = build_qrels_dict(test, id_map)
        log_query_results_compact(test.query_text, test.expected_result_ids, results, reverse_map, top_k)
        run_dict[test.query_id] = build_run_dict_entry(results)

    # Log negative test summary
    total_negative = negative_tests_passed + negative_tests_failed
    if total_negative > 0:
        print(f"\n  📋 Negative tests: {negative_tests_passed}/{total_negative} passed")

    if not qrels_dict or not run_dict:
        print("⚠️  No valid qrels or run data to evaluate")
        return None

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)  # type: ignore

    ranx_metrics = evaluate(qrels, run, EvaluationMetrics.metrics)  # type: ignore
    metrics = EvaluationMetrics.from_ranx_result(ranx_metrics)

    print(f"\n📊 Metrics for {use_case.title}:")
    print(metrics)

    return UseCaseResult(
        use_case_title=use_case.title,
        metrics=metrics,
        num_queries=len(qrels_dict),
    )


async def run_evaluation(
    nevis_client: NevisClient,
    suite: EvalSuite,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """
    Run a complete evaluation suite against the Nevis API.

    This is the main entry point for running evaluations. It:
    1. Sets up the corpus (creates clients and documents)
    2. Waits for document processing
    3. Runs all use cases
    4. Collects and returns results

    Args:
        nevis_client: API client connected to the Nevis instance
        suite: Evaluation suite containing corpus and test cases
        config: Optional configuration (defaults to EvaluationConfig())

    Returns:
        EvaluationResult with all metrics and failures
    """
    if config is None:
        config = EvaluationConfig()

    print(f"\n{'='*60}")
    print(f"Starting Evaluation Suite: {suite.suite_name}")
    print(f"{'='*60}")

    # Setup corpus
    print(f"\n{'='*60}")
    print("Setting up corpus...")
    print(f"{'='*60}")

    id_map = await setup_corpus(suite, nevis_client)
    reverse_map = {str(uuid_val): synthetic_id for synthetic_id, uuid_val in id_map.items()}
    print(f"✅ Corpus setup complete: {len(id_map)} entities created")

    # Run all use cases
    all_results: list[UseCaseResult] = []
    failures: list[tuple[str, str]] = []

    for use_case in suite.use_cases:
        try:
            result = await run_eval_use_case(
                use_case=use_case,
                id_map=id_map,
                reverse_map=reverse_map,
                nevis_client=nevis_client,
                top_k=config.top_k,
            )

            if result:
                all_results.append(result)

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

    # Print summary
    if config.verbose:
        EvaluationReporter.print_summary(
            suite_name=suite.suite_name,
            total_use_cases=len(suite.use_cases),
            all_results=all_results,
            failures=failures,
        )

    return EvaluationResult(
        suite_name=suite.suite_name,
        total_use_cases=len(suite.use_cases),
        results=all_results,
        failures=failures,
    )


async def run_evaluation_from_file(
    nevis_client: NevisClient,
    data_path: Path,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """
    Run evaluation from a JSON file.

    Convenience wrapper that loads the evaluation suite from a file.

    Args:
        nevis_client: API client connected to the Nevis instance
        data_path: Path to the JSON evaluation suite file
        config: Optional configuration

    Returns:
        EvaluationResult with all metrics and failures
    """
    suite = load_eval_suite(data_path)
    return await run_evaluation(nevis_client, suite, config)
