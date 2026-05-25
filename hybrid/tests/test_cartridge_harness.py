from __future__ import annotations

import sys
from pathlib import Path

DEEPSEEK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEEPSEEK.parent))

from hybrid.cartridge_harness import (
    ExactFirstLineScorer,
    TaskExample,
    build_private_fact_tasks,
    build_summary,
    evaluate_text_runner,
    normalize_first_line,
)
from hybrid.cartridge_harness.core import compare_rows


def test_private_fact_suite_has_train_and_heldout_paraphrases():
    tasks = build_private_fact_tasks()

    assert len([task for task in tasks if task.split == "train"]) == 36
    assert len([task for task in tasks if task.split == "heldout"]) == 24
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert all(task.metadata["name"].startswith("Project ") for task in tasks)


def test_exact_first_line_scorer_normalizes_only_first_line():
    scorer = ExactFirstLineScorer()

    ok, generated, expected = scorer(" raven-041.\nextra", "RAVEN-041")

    assert ok
    assert generated == expected == "RAVEN-041"
    assert normalize_first_line("MICA 772") == "MICA772"


def test_evaluate_text_runner_and_summary_counts_by_split():
    tasks = [
        TaskExample("a", "train", "prompt a", "YES"),
        TaskExample("b", "heldout", "prompt b", "NO"),
    ]
    answers = {"prompt a": "YES", "prompt b": "maybe"}

    rows = evaluate_text_runner(tasks, lambda prompt: answers[prompt])
    summary = build_summary(rows)

    assert summary.total == 2
    assert summary.correct == 1
    assert summary.by_split["train"] == {"correct": 1, "total": 1}
    assert summary.by_split["heldout"] == {"correct": 0, "total": 1}


def test_compare_rows_finds_improvements_and_regressions():
    tasks = [
        TaskExample("a", "train", "prompt a", "YES"),
        TaskExample("b", "heldout", "prompt b", "NO"),
    ]
    baseline = evaluate_text_runner(tasks, lambda prompt: "wrong" if prompt == "prompt a" else "NO")
    cartridge = evaluate_text_runner(tasks, lambda prompt: "YES" if prompt == "prompt a" else "wrong")

    comparison = compare_rows(baseline, cartridge)

    assert [row["task_id"] for row in comparison["improved"]] == ["a"]
    assert [row["task_id"] for row in comparison["regressed"]] == ["b"]