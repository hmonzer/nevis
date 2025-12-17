"""Evaluation runner - orchestrates the evaluation process."""

from dataclasses import dataclass
from pathlib import Path

from ranx import Qrels, Run, evaluate

from src.client import NevisClient
from src.eval.data_setup import CorpusSetup, load_eval_suite
from src.eval.metrics import EvaluationMetrics, EvaluationResult, UseCaseResult
from src.eval.reporter import EvaluationReporter
from src.eval.schema import EvalSuite, UseCase, TestItem, Corpus


@dataclass
class EvaluationConfig:
    """Configuration for running an evaluation."""

    top_k: int = 5
    verbose: bool = True


class EvalRunner:
    """
    Orchestrates the evaluation process.

    Separates concerns:
    - Corpus setup: handled by CorpusSetup
    - Query execution: handled internally
    - Metrics calculation: uses ranx library
    - Reporting: handled by EvaluationReporter
    """

    def __init__(self, nevis_client: NevisClient):
        """
        Initialize the evaluation runner.

        Args:
            nevis_client: API client for search requests
        """
        self.nevis_client = nevis_client
        self.corpus_setup = CorpusSetup(nevis_client)
        self._corpus: Corpus | None = None

    async def run_from_file(
        self,
        data_path: Path,
        config: EvaluationConfig | None = None,
    ) -> EvaluationResult:
        """
        Run evaluation from a JSON file.

        Args:
            data_path: Path to the JSON evaluation suite file
            config: Optional configuration

        Returns:
            EvaluationResult with all metrics and failures
        """
        # Load input file
        suite = load_eval_suite(data_path)
        # Set up data in Nevis
        self._corpus = await self.corpus_setup.setup(suite)

        # Run evaluation
        return await self.run_suite(suite, config)

    async def run_suite(
        self,
        suite: EvalSuite,
        config: EvaluationConfig | None = None,
    ) -> EvaluationResult:
        """
        Run a complete evaluation suite.

        Args:
            suite: Evaluation suite containing corpus and test cases
            config: Optional configuration

        Returns:
            EvaluationResult with all metrics and failures
        """
        if config is None:
            config = EvaluationConfig()

        # Run all use cases
        results, failures = await self._run_all_use_cases(suite.use_cases, config)

        # Build result
        result = EvaluationResult(
            suite_name=suite.suite_name,
            total_use_cases=len(suite.use_cases),
            results=results,
            failures=failures,
        )

        # Print summary
        if config.verbose:
            EvaluationReporter.print_summary(result)

        return result

    async def _setup_corpus(self, suite: EvalSuite) -> None:
        """Setup corpus and populate nevis_ids on records."""
        print(f"\n{'='*60}")
        print("Setting up corpus...")
        print(f"{'='*60}")

        corpus_setup = CorpusSetup(self.nevis_client)
        self._corpus = await corpus_setup.setup(suite)

        print(f"✅ Corpus setup complete: {self._corpus.entity_count} entities created")

    async def _run_all_use_cases(
        self,
        use_cases: list[UseCase],
        config: EvaluationConfig,
    ) -> tuple[list[UseCaseResult], list[tuple[str, str]]]:
        """Run all use cases and collect results."""
        results: list[UseCaseResult] = []
        failures: list[tuple[str, str]] = []

        for use_case in use_cases:
            try:
                result = await self._run_use_case(use_case, config.top_k)

                if result:
                    results.append(result)
                    if result.metrics.recall_at_5 <= 0:
                        failures.append((
                            use_case.title,
                            f"Recall@5 is {result.metrics.recall_at_5:.4f}, expected > 0",
                        ))
                else:
                    failures.append((use_case.title, "No valid evaluation data"))

            except Exception as e:
                print(f"❌ Error running use case '{use_case.title}': {e}")
                failures.append((use_case.title, f"Exception: {str(e)}"))

        return results, failures

    async def _run_use_case(
        self,
        use_case: UseCase,
        top_k: int,
    ) -> UseCaseResult | None:
        """Run evaluation for a single use case."""
        assert self._corpus is not None

        EvaluationReporter.print_use_case_header(use_case.title)

        qrels_dict: dict[str, dict[str, int]] = {}
        run_dict: dict[str, dict[str, float]] = {}

        negative_tests_passed = 0
        negative_tests_failed = 0

        for test in use_case.tests:
            # Execute query and record results in TestItem
            await self._execute_test(test, top_k)

            # Handle negative test cases
            if test.is_negative_test:
                assert test.result is not None
                if test.result.total_results == 0:
                    negative_tests_passed += 1
                    EvaluationReporter.print_negative_test_result(passed=True)
                else:
                    negative_tests_failed += 1
                    EvaluationReporter.print_negative_test_result(
                        passed=False, result_count=test.result.total_results
                    )
                    qrels_dict[test.query_id] = {"__nonexistent_doc__": 1}
                    run_dict[test.query_id] = test.result.run_entry
                continue

            # Normal case - build qrels using corpus lookups
            assert test.result is not None
            qrels_dict[test.query_id] = test.build_qrels(self._corpus)
            run_dict[test.query_id] = test.result.run_entry

            # Log results using TestItem's comparison methods
            EvaluationReporter.log_test_results(test, top_k)

        EvaluationReporter.print_negative_test_summary(
            negative_tests_passed,
            negative_tests_passed + negative_tests_failed,
        )

        if not qrels_dict or not run_dict:
            print("⚠️  No valid qrels or run data to evaluate")
            return None

        # Calculate metrics using ranx
        qrels = Qrels(qrels_dict)
        run = Run(run_dict)  # type: ignore

        ranx_metrics = evaluate(qrels, run, EvaluationMetrics.metrics)  # type: ignore
        metrics = EvaluationMetrics.from_ranx_result(ranx_metrics)

        EvaluationReporter.print_metrics(use_case.title, metrics)

        return UseCaseResult(
            use_case_title=use_case.title,
            metrics=metrics,
            num_queries=len(qrels_dict),
        )

    async def _execute_test(self, test: TestItem, top_k: int) -> None:
        """Execute a test query and store results in the TestItem."""
        assert self._corpus is not None

        results = await self.nevis_client.search(query=test.query_text, top_k=top_k)

        # Record results in TestItem using corpus for ID lookup
        test.record_results(results, self._corpus)

        # Print query summary
        assert test.result is not None
        EvaluationReporter.print_query_summary(
            query_text=test.query_text,
            total_results=test.result.total_results,
            num_clients=test.result.num_clients,
            num_documents=test.result.num_documents,
        )
