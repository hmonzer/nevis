"""CLI for running search evaluation against a Nevis API instance."""

import argparse
import asyncio
import sys
from pathlib import Path

from src.client import NevisClient
from src.eval.runner import EvalRunner, EvaluationConfig


def get_default_data_path() -> Path:
    """Get the default evaluation data file path."""
    # Look in tests/e2e_eval/data for backwards compatibility
    return Path(__file__).parent.parent.parent / "tests" / "e2e_eval" / "data" / "synthetic_wealth_data.json"


async def run_evaluation(args: argparse.Namespace) -> int:
    """
    Run the evaluation.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    data_path = Path(args.data) if args.data else get_default_data_path()

    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        return 1

    print(f"🚀 Nevis Search Evaluation Runner")
    print(f"   API URL: {args.url}")
    print(f"   Data file: {data_path}")
    print(f"   Top-K: {args.top_k}")
    print()

    config = EvaluationConfig(
        top_k=args.top_k,
        verbose=True,
    )

    async with NevisClient(base_url=args.url) as client:
        try:
            runner = EvalRunner(client)
            result = await runner.run_from_file(data_path, config)

            if len(result.results) == 0:
                print("\n❌ Evaluation failed: No use cases succeeded")
                return 1

            if result.failures:
                print(f"\n⚠️  Evaluation completed with {len(result.failures)} failure(s)")
                return 1

            print(f"\n✅ Evaluation completed successfully!")
            avg = result.average_metrics
            if avg:
                print(f"   Average MRR: {avg.mrr:.4f}")
                print(f"   Average Recall@5: {avg.recall_at_5:.4f}")
                print(f"   Average NDCG@5: {avg.ndcg_at_5:.4f}")
                print(f"   Average Precision: {avg.precision:.4f}")
            return 0

        except Exception as e:
            print(f"\n❌ Evaluation failed with error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run search evaluation against a Nevis API instance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run against local docker-compose
  python -m src.eval --url http://localhost:8000

  # Run with custom data file
  python -m src.eval --url http://localhost:8000 --data my_eval_data.json

  # Run with higher top-k
  python -m src.eval --url http://localhost:8000 --top-k 10
        """,
    )

    parser.add_argument(
        "--url",
        required=True,
        help="Base URL of the Nevis API (e.g., http://localhost:8000)",
    )

    parser.add_argument(
        "--data",
        help="Path to the evaluation data JSON file (default: synthetic_wealth_data.json)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve per query (default: 5)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with full stack traces",
    )

    return parser


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    return asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    sys.exit(main())
