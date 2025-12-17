"""Evaluation metrics models and calculations."""

from typing import Any, ClassVar

from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Information retrieval metrics for a single use case."""

    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    recall_at_5: float = Field(..., ge=0.0, le=1.0, description="Recall at 5 results")
    ndcg_at_5: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized Discounted Cumulative Gain at 5"
    )
    precision: float = Field(
        0.0, ge=0.0, le=1.0, description="Precision of results"
    )

    # Metric names for ranx.evaluate() - ClassVar to avoid Pydantic field treatment
    metrics: ClassVar[list[str]] = ["mrr", "recall@5", "ndcg@5", "precision"]

    @classmethod
    def from_ranx_result(cls, metrics: Any) -> "EvaluationMetrics":
        """Create EvaluationMetrics from ranx evaluate() result."""
        return cls(
            mrr=metrics["mrr"],
            recall_at_5=metrics["recall@5"],
            ndcg_at_5=metrics["ndcg@5"],
            precision=metrics["precision"],
        )

    def __str__(self) -> str:
        """Format metrics for display."""
        return (
            f"MRR: {self.mrr:.4f}, Recall@5: {self.recall_at_5:.4f}, "
            f"NDCG@5: {self.ndcg_at_5:.4f}, Precision: {self.precision:.4f}"
        )


class UseCaseResult(BaseModel):
    """Result for a single use case evaluation."""

    use_case_title: str = Field(..., description="Name of the use case")
    metrics: EvaluationMetrics = Field(..., description="Evaluation metrics")
    num_queries: int = Field(..., gt=0, description="Number of queries evaluated")

    def __str__(self) -> str:
        """Format use case result for display."""
        return f"{self.use_case_title}: {self.metrics}"


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across multiple use cases."""

    mrr: float = Field(..., ge=0.0, le=1.0)
    recall_at_5: float = Field(..., ge=0.0, le=1.0)
    ndcg_at_5: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)

    @classmethod
    def from_results(cls, results: list[UseCaseResult]) -> "AggregatedMetrics | None":
        """Calculate average metrics from a list of use case results."""
        if not results:
            return None

        return cls(
            mrr=sum(r.metrics.mrr for r in results) / len(results),
            recall_at_5=sum(r.metrics.recall_at_5 for r in results) / len(results),
            ndcg_at_5=sum(r.metrics.ndcg_at_5 for r in results) / len(results),
            precision=sum(r.metrics.precision for r in results) / len(results),
        )


class EvaluationResult(BaseModel):
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
    def average_metrics(self) -> AggregatedMetrics | None:
        """Calculate average metrics across all successful use cases."""
        return AggregatedMetrics.from_results(self.results)

    @property
    def has_failures(self) -> bool:
        """Check if there were any failures."""
        return len(self.failures) > 0

    @property
    def all_passed(self) -> bool:
        """Check if all use cases passed."""
        return len(self.results) == self.total_use_cases and not self.failures
