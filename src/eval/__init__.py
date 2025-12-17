"""
Evaluation module for search quality assessment.

This module provides tools for evaluating search quality using standard IR metrics.
It can be used in two ways:

1. Via pytest tests:
   ```python
   from src.eval import EvalRunner, EvaluationConfig
   result = await EvalRunner(nevis_client).run_suite(suite, config)
   ```

2. Via command line:
   ```bash
   python -m src.eval --url http://localhost:8000 --data eval_data.json
   ```
"""

from src.eval.schema import (
    EvalSuite,
    Corpus,
    CorpusRecord,
    UseCase,
    TestItem,
    TestResult,
    TestComparison,
    RetrievedResult,
    ClientRecord,
    DocumentRecord,
)
from src.eval.metrics import (
    EvaluationMetrics,
    UseCaseResult,
    EvaluationResult,
)
from src.eval.reporter import EvaluationReporter
from src.eval.data_setup import CorpusSetup
from src.eval.runner import EvalRunner, EvaluationConfig

__all__ = [
    # Schema
    "EvalSuite",
    "Corpus",
    "CorpusRecord",
    "UseCase",
    "TestItem",
    "TestResult",
    "TestComparison",
    "RetrievedResult",
    "ClientRecord",
    "DocumentRecord",
    # Metrics
    "EvaluationMetrics",
    "UseCaseResult",
    "EvaluationResult",
    # Reporter
    "EvaluationReporter",
    # Data Setup
    "CorpusSetup",
    # Runner
    "EvalRunner",
    "EvaluationConfig",
]
