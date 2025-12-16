"""Evaluation models and reporting for search evaluation."""
from typing import Any
from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Encapsulates information retrieval metrics for a single use case."""

    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    recall_at_5: float = Field(..., ge=0.0, le=1.0, description="Recall at 5 results")
    ndcg_at_5: float = Field(..., ge=0.0, le=1.0, description="Normalized Discounted Cumulative Gain at 5")
    precision_at_5: float = Field(0.0, ge=0.0, le=1.0, description="Precision at 5 results (default 0.0)")

    @classmethod
    def from_ranx_result(cls, metrics: Any) -> "EvaluationMetrics":
        """
        Create EvaluationMetrics from ranx evaluate() result.

        Args:
            metrics: Dictionary-like result from ranx.evaluate()

        Returns:
            EvaluationMetrics instance
        """
        return cls(
            mrr=metrics["mrr"],  # type: ignore
            recall_at_5=metrics["recall@5"],  # type: ignore
            ndcg_at_5=metrics["ndcg@5"],  # type: ignore
            precision_at_5=metrics["precision@5"]
        )

    def __str__(self) -> str:
        """Format metrics for display."""
        return f"MRR: {self.mrr:.4f}, Recall@5: {self.recall_at_5:.4f}, NDCG@5: {self.ndcg_at_5:.4f}, Precision@5: {self.precision_at_5:.4f}"


class UseCaseResult(BaseModel):
    """Result for a single use case evaluation."""

    use_case_title: str = Field(..., description="Name of the use case")
    metrics: EvaluationMetrics = Field(..., description="Evaluation metrics")
    num_queries: int = Field(..., gt=0, description="Number of queries evaluated")

    def __str__(self) -> str:
        """Format use case result for display."""
        return f"{self.use_case_title}: {self.metrics}"


class EvaluationReporter:
    """Handles formatting and printing of evaluation results."""

    @staticmethod
    def print_summary(
        suite_name: str,
        total_use_cases: int,
        all_results: list[UseCaseResult],
        failures: list[tuple[str, str]],
    ) -> None:
        """
        Print comprehensive evaluation summary with formatted table.

        Args:
            suite_name: Name of the evaluation suite
            total_use_cases: Total number of use cases
            all_results: List of successfully evaluated use cases
            failures: List of (use_case_title, failure_reason) tuples
        """
        # Print header
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Suite: {suite_name}")
        print(f"Total Use Cases: {total_use_cases}")
        print(f"Successfully Evaluated: {len(all_results)}")
        print(f"Failures: {len(failures)}")

        # Print metrics table if we have results
        if all_results:
            EvaluationReporter._print_metrics_table(all_results)

        # Print failures if any
        if failures:
            EvaluationReporter._print_failures(failures)

    @staticmethod
    def _print_metrics_table(all_results: list[UseCaseResult]) -> None:
        """
        Print formatted table of metrics by use case.

        Args:
            all_results: List of use case results
        """
        print("\n📊 METRICS BY USE CASE")
        print(f"{'='*80}")

        # Calculate column widths
        max_name_width = max(len(result.use_case_title) for result in all_results)
        max_name_width = max(max_name_width, len("Use Case"))
        max_name_width = min(max_name_width, 40)  # Cap at 40 chars

        # Print table header
        header = f"{'Use Case':<{max_name_width}} | {'MRR':>8} | {'Recall@5':>10} | {'NDCG@5':>8} | {'Precision@5':>8}"
        print(header)
        print("-" * len(header))

        # Print each use case metrics
        for result in all_results:
            # Truncate long titles
            display_title = (
                result.use_case_title
                if len(result.use_case_title) <= max_name_width
                else result.use_case_title[:max_name_width-3] + "..."
            )

            print(
                f"{display_title:<{max_name_width}} | "
                f"{result.metrics.mrr:>8.4f} | "
                f"{result.metrics.recall_at_5:>10.4f} | "
                f"{result.metrics.ndcg_at_5:>8.4f} | "
                f"{result.metrics.precision_at_5:>8.4f}"
            )

        # Print separator before average
        print("-" * len(header))

        # Calculate and print averages
        avg_mrr = sum(r.metrics.mrr for r in all_results) / len(all_results)
        avg_recall = sum(r.metrics.recall_at_5 for r in all_results) / len(all_results)
        avg_ndcg = sum(r.metrics.ndcg_at_5 for r in all_results) / len(all_results)
        avg_precision = sum(r.metrics.precision_at_5 for r in all_results) / len(all_results)

        print(f"{'AVERAGE':<{max_name_width}} | {avg_mrr:>8.4f} | {avg_recall:>10.4f} | {avg_ndcg:>8.4f} | {avg_precision:>8.4f}")
        print("=" * len(header))

    @staticmethod
    def _print_failures(failures: list[tuple[str, str]]) -> None:
        """
        Print failed use cases with reasons.

        Args:
            failures: List of (use_case_title, failure_reason) tuples
        """
        print("\n⚠️  FAILED USE CASES")
        print(f"{'='*80}")
        for title, reason in failures:
            print(f"  • {title}")
            print(f"    Reason: {reason}")
