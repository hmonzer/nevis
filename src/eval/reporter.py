"""Evaluation result reporting and formatting."""

from typing import Any, TYPE_CHECKING

from src.eval.metrics import UseCaseResult, EvaluationResult

if TYPE_CHECKING:
    from src.eval.schema import TestItem


class EvaluationReporter:
    """Handles formatting and printing of evaluation results."""

    @staticmethod
    def print_summary(result: EvaluationResult) -> None:
        """
        Print comprehensive evaluation summary with formatted table.

        Args:
            result: Complete evaluation result
        """
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Suite: {result.suite_name}")
        print(f"Total Use Cases: {result.total_use_cases}")
        print(f"Successfully Evaluated: {len(result.results)}")
        print(f"Failures: {len(result.failures)}")

        if result.results:
            EvaluationReporter._print_metrics_table(result.results)

        if result.failures:
            EvaluationReporter._print_failures(result.failures)

    @staticmethod
    def _print_metrics_table(all_results: list[UseCaseResult]) -> None:
        """Print formatted table of metrics by use case."""
        print("\n📊 METRICS BY USE CASE")
        print(f"{'='*80}")

        max_name_width = max(len(r.use_case_title) for r in all_results)
        max_name_width = max(max_name_width, len("Use Case"))
        max_name_width = min(max_name_width, 40)

        header = (
            f"{'Use Case':<{max_name_width}} | "
            f"{'MRR':>8} | {'Recall@5':>10} | {'NDCG@5':>8} | {'Precision':>9}"
        )
        print(header)
        print("-" * len(header))

        for result in all_results:
            display_title = (
                result.use_case_title
                if len(result.use_case_title) <= max_name_width
                else result.use_case_title[: max_name_width - 3] + "..."
            )

            print(
                f"{display_title:<{max_name_width}} | "
                f"{result.metrics.mrr:>8.4f} | "
                f"{result.metrics.recall_at_5:>10.4f} | "
                f"{result.metrics.ndcg_at_5:>8.4f} | "
                f"{result.metrics.precision:>9.4f}"
            )

        print("-" * len(header))

        # Calculate and print averages
        avg_mrr = sum(r.metrics.mrr for r in all_results) / len(all_results)
        avg_recall = sum(r.metrics.recall_at_5 for r in all_results) / len(all_results)
        avg_ndcg = sum(r.metrics.ndcg_at_5 for r in all_results) / len(all_results)
        avg_precision = sum(r.metrics.precision for r in all_results) / len(all_results)

        print(
            f"{'AVERAGE':<{max_name_width}} | "
            f"{avg_mrr:>8.4f} | {avg_recall:>10.4f} | "
            f"{avg_ndcg:>8.4f} | {avg_precision:>9.4f}"
        )
        print("=" * len(header))

    @staticmethod
    def _print_failures(failures: list[tuple[str, str]]) -> None:
        """Print failed use cases with reasons."""
        print("\n⚠️  FAILED USE CASES")
        print(f"{'='*80}")
        for title, reason in failures:
            print(f"  • {title}")
            print(f"    Reason: {reason}")

    @staticmethod
    def log_test_results(test: "TestItem", top_k: int = 5) -> None:
        """
        Log compact comparison of expected vs retrieved results for a TestItem.

        Only logs if there are issues (missing or extra results).

        Args:
            test: TestItem with recorded results
            top_k: Number of top results to consider
        """
        if not test.has_result or not test.has_issues(top_k):
            return

        assert test.result is not None
        comparison = test.get_comparison(top_k)

        # Format retrieved results with scores
        retrieved_with_scores = [
            r.format_with_score() for r in test.result.retrieved[:top_k]
        ]

        print(f"    ⚠️  Query: '{test.query_text[:60]}'")
        print(f"        Expected: {', '.join(test.expected_result_ids)}")
        print(f"        Retrieved (top {top_k}): {', '.join(retrieved_with_scores)}")
        print(f"        ✅ Correct: {comparison.precision_str}", end="")

        if comparison.missing:
            print(f" | ❌ Missing: {', '.join(comparison.missing)}", end="")

        if comparison.extra:
            extra_details = test.get_extra_results_details(top_k)
            print(f" | 🔴 Extra: {', '.join(extra_details)}", end="")

        print()

    @staticmethod
    def print_use_case_header(title: str) -> None:
        """Print use case section header."""
        print(f"\n{'='*60}")
        print(f"Running Use Case: {title}")
        print(f"{'='*60}")

    @staticmethod
    def print_query_summary(
        query_text: str,
        total_results: int,
        num_clients: int,
        num_documents: int,
    ) -> None:
        """Print summary of query results."""
        print(
            f"  Query '{query_text[:50]}...': {total_results} total results "
            f"({num_clients} clients, {num_documents} documents)"
        )

    @staticmethod
    def print_negative_test_result(passed: bool, result_count: int = 0) -> None:
        """Print result of a negative test case."""
        if passed:
            print(f"    ✅ Negative test passed: correctly returned 0 results")
        else:
            print(
                f"    ❌ Negative test failed: expected 0 results, got {result_count}"
            )

    @staticmethod
    def print_negative_test_summary(passed: int, total: int) -> None:
        """Print summary of negative tests."""
        if total > 0:
            print(f"\n  📋 Negative tests: {passed}/{total} passed")

    @staticmethod
    def print_metrics(title: str, metrics: Any) -> None:
        """Print metrics for a use case."""
        print(f"\n📊 Metrics for {title}:")
        print(metrics)
