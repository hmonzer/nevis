"""
E2E evaluation tests for wealth manager search functionality.

Uses the src.eval module which can also be invoked via CLI:
    python -m src.eval --url http://localhost:8000

To run a specific evaluation dataset:
    pytest tests/e2e_eval/test_wealth_manager_eval.py -k "synthetic_wealth_data" -v -s
"""
from pathlib import Path

import pytest

from src.eval import EvaluationConfig

# Directory containing evaluation data files
DATA_DIR = Path(__file__).parent / "data"

# List of evaluation datasets to run
# Add new JSON files here to include them in the test suite
EVAL_DATASETS = [
    "synthetic_wealth_data.json",
    "synthetic_wealth_data_large.json",
    # Add more datasets here as they are created:
]


def get_dataset_id(dataset_path: str) -> str:
    """Extract a readable test ID from the dataset filename."""
    return Path(dataset_path).stem


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dataset_file",
    EVAL_DATASETS,
    ids=get_dataset_id,
)
async def test_wealth_manager_eval(caplog, eval_runner, dataset_file: str):
    """
    E2E evaluation of search API using synthetic wealth management data.

    This parametrized test runs against each dataset in EVAL_DATASETS, allowing
    multiple evaluation sets to be tested with the same framework.

    Args:
        eval_runner: EvalRunner fixture configured with test client
        dataset_file: Name of the JSON dataset file to evaluate

    The test:
    1. Loads an evaluation suite from the specified JSON file
    2. Ingests all corpus data once (clients and documents)
    3. For each use case, runs queries via the search API endpoint
    4. Compares search results against expected results using IR metrics
    5. Asserts that at least one use case succeeded
    """

    # Raise log level just to keep summary output clean. For debugging, comment the below line.
    caplog.set_level("ERROR")

    # Load the evaluation suite
    data_path = DATA_DIR / dataset_file
    if not data_path.exists():
        pytest.skip(f"Dataset file not found: {data_path}")

    # Run evaluation using the eval runner fixture
    config = EvaluationConfig(top_k=5, verbose=True)
    result = await eval_runner.run_from_file(data_path, config)

    # Assertions
    assert len(result.results) > 0, f"All use cases failed for {dataset_file} - no valid metrics collected"