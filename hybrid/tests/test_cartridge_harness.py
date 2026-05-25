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
from hybrid.cartridge_harness.rack_builder import assemble_rack_summary
from hybrid.cartridge_harness.suites import build_all_suites, get_suite


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


def test_rack_suites_have_unique_ids_and_split_tasks():
    suites = build_all_suites()

    assert [suite.suite_id for suite in suites] == [
        "private_facts",
        "arithmetic",
        "code_labels",
        "safety_labels",
        "instruction_format",
    ]
    assert len({suite.cartridge_id for suite in suites}) == len(suites)
    for suite in suites:
        assert get_suite(suite.suite_id) == suite
        assert any(task.split == "train" for task in suite.tasks)
        assert any(task.split == "heldout" for task in suite.tasks)


def test_assemble_rack_summary_from_suite_outputs(tmp_path: Path):
    suite = get_suite("arithmetic")
    suite_dir = tmp_path / suite.suite_id
    suite_dir.mkdir()
    (suite_dir / "summary.json").write_text(
        """
        {
          "artifact": "artifacts/rack/arithmetic/cartridge_best.pt",
          "baseline_summary": {"total": 2, "correct": 0, "accuracy": 0.0, "by_split": {}},
          "cartridge_summary": {"total": 2, "correct": 2, "accuracy": 1.0, "by_split": {}},
          "improved": [{"task_id": "a"}],
          "regressed": []
        }
        """,
        encoding="utf-8",
    )

    summary = assemble_rack_summary(
        model="Qwen/Qwen2.5-1.5B",
        device="cuda",
        out_dir=tmp_path,
        suites=["arithmetic"],
    )

    assert summary["items"][0]["suite"]["suite_id"] == "arithmetic"
    assert summary["items"][0]["improved_count"] == 1
    assert (tmp_path / "rack_manifest.json").exists()
    assert (tmp_path / "rack_summary.json").exists()