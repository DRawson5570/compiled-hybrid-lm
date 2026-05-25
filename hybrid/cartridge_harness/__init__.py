"""Owned harness for cartridge research and self-improvement loops."""

from hybrid.cartridge_harness.core import (
    EvalRow,
    EvalSummary,
    ExactFirstLineScorer,
    TaskExample,
    build_summary,
    evaluate_text_runner,
    normalize_first_line,
)
from hybrid.cartridge_harness.private_facts import (
    PRIVATE_FACTS,
    build_private_fact_tasks,
)

__all__ = [
    "EvalRow",
    "EvalSummary",
    "ExactFirstLineScorer",
    "PRIVATE_FACTS",
    "TaskExample",
    "build_private_fact_tasks",
    "build_summary",
    "evaluate_text_runner",
    "normalize_first_line",
]