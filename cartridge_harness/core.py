"""Small, owned evaluation primitives for cartridge-building experiments.

The harness keeps benchmark mechanics separate from any one external repo. A
research loop supplies tasks, a text generator, and a scorer; the harness records
baseline rows, cartridge rows, improvements, regressions, and split metrics.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable


TextGenerator = Callable[[str], str]


@dataclass(frozen=True)
class TaskExample:
    """One prompt/answer item in a cartridge experiment."""

    task_id: str
    split: str
    prompt: str
    expected: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalRow:
    """A generated answer plus score for one task."""

    task_id: str
    split: str
    prompt: str
    expected: str
    generated: str
    normalized_generated: str
    normalized_expected: str
    correct: bool
    metadata: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate score counts for an evaluation run."""

    total: int
    correct: int
    by_split: dict[str, dict[str, int]]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_json(self) -> dict:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "by_split": self.by_split,
        }


def normalize_first_line(text: str) -> str:
    """Normalize the first generated line for exact-answer scoring."""

    first = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"[^A-Za-z0-9-]", "", first).upper()


class ExactFirstLineScorer:
    """Strict scorer for prompt-specific facts and answer-only capabilities."""

    def __call__(self, generated: str, expected: str) -> tuple[bool, str, str]:
        normalized_generated = normalize_first_line(generated)
        normalized_expected = normalize_first_line(expected)
        return (
            normalized_generated == normalized_expected,
            normalized_generated,
            normalized_expected,
        )


def evaluate_text_runner(
    tasks: Iterable[TaskExample],
    generator: TextGenerator,
    scorer: ExactFirstLineScorer | Callable[[str, str], tuple[bool, str, str]] | None = None,
) -> list[EvalRow]:
    """Run a text generator across tasks and return row-level scores."""

    score = scorer or ExactFirstLineScorer()
    rows: list[EvalRow] = []
    for task in tasks:
        generated = generator(task.prompt)
        correct, normalized_generated, normalized_expected = score(generated, task.expected)
        rows.append(
            EvalRow(
                task_id=task.task_id,
                split=task.split,
                prompt=task.prompt,
                expected=task.expected,
                generated=generated,
                normalized_generated=normalized_generated,
                normalized_expected=normalized_expected,
                correct=correct,
                metadata=dict(task.metadata),
            )
        )
    return rows


def build_summary(rows: Iterable[EvalRow]) -> EvalSummary:
    """Build aggregate counts by split and overall."""

    materialized = list(rows)
    by_split: dict[str, dict[str, int]] = {}
    for row in materialized:
        split = by_split.setdefault(row.split, {"correct": 0, "total": 0})
        split["total"] += 1
        split["correct"] += int(row.correct)
    return EvalSummary(
        total=len(materialized),
        correct=sum(int(row.correct) for row in materialized),
        by_split=by_split,
    )


def compare_rows(baseline_rows: Iterable[EvalRow], cartridge_rows: Iterable[EvalRow]) -> dict[str, list[dict]]:
    """Find fail-to-pass improvements and pass-to-fail regressions."""

    baseline_by_id = {row.task_id: row for row in baseline_rows}
    cartridge_by_id = {row.task_id: row for row in cartridge_rows}
    improved: list[dict] = []
    regressed: list[dict] = []
    for task_id, baseline in baseline_by_id.items():
        cartridge = cartridge_by_id.get(task_id)
        if cartridge is None:
            continue
        item = {
            "task_id": task_id,
            "split": baseline.split,
            "expected": baseline.expected,
            "baseline": baseline.generated,
            "baseline_first": baseline.normalized_generated,
            "cartridge": cartridge.generated,
            "cartridge_first": cartridge.normalized_generated,
            "metadata": baseline.metadata,
        }
        if not baseline.correct and cartridge.correct:
            improved.append(item)
        elif baseline.correct and not cartridge.correct:
            regressed.append(item)
    return {"improved": improved, "regressed": regressed}