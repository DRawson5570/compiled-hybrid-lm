"""Tests for ARC benchmark harness.

These tests use toy data and do not require downloading ARC or loading any model.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_staging_root = Path(__file__).resolve().parent.parent
_deepseek = Path("/home/drawson/deepseek_experiments")
for p in [str(_staging_root), str(_deepseek)]:
    if p not in sys.path:
        sys.path.insert(0, str(p))

from hybrid.benchmarks.arc_data import (
    ARCExample,
    ARCChoice,
    normalize_arc_example,
    load_arc_dataset,
)
from hybrid.benchmarks.arc_prompts import get_template
from hybrid.benchmarks.arc_scoring import ChoiceScore, ScoredExample
from hybrid.benchmarks.arc_reports import write_reports


def _make_arc_row(question="What is 2+2?", choices=None, answer_key="A"):
    if choices is None:
        choices = [
            {"label": "A", "text": "4"},
            {"label": "B", "text": "22"},
            {"label": "C", "text": "0"},
            {"label": "D", "text": "2"},
        ]
    return {"id": "test_001", "question": question, "choices": choices, "answerKey": answer_key}


class TestNormalizeArcExample:
    def test_letters(self):
        row = _make_arc_row()
        ex = normalize_arc_example(row, "validation", "ARC-Challenge")
        assert ex.id == "test_001"
        assert ex.question == "What is 2+2?"
        assert len(ex.choices) == 4
        assert ex.choices[0].label == "A"
        assert ex.choices[0].text == "4"
        assert ex.answer_key == "A"
        assert ex.split == "validation"
        assert ex.config == "ARC-Challenge"

    def test_numeric_labels(self):
        row = {
            "id": "num_001",
            "question": "Pick one.",
            "choices": [
                {"label": "1", "text": "first"},
                {"label": "2", "text": "second"},
            ],
            "answerKey": 2,
        }
        ex = normalize_arc_example(row, "test", "ARC-Easy")
        assert ex.choices[0].label == "1"
        assert ex.answer_key == "2"

    def test_none_answer_key(self):
        row = _make_arc_row()
        row.pop("answerKey")
        ex = normalize_arc_example(row, "test", "ARC-Easy")
        assert ex.answer_key is None


class TestArcExampleValidation:
    def test_invalid_answer_key_rejected(self):
        row = _make_arc_row(answer_key="Z")
        ex = normalize_arc_example(row, "validation", "ARC-Challenge")
        issues = ex.is_valid()
        assert len(issues) > 0
        assert any("Z" in issue for issue in issues)

    def test_empty_question(self):
        row = _make_arc_row(question="")
        ex = normalize_arc_example(row, "validation", "ARC-Challenge")
        assert ex.is_valid()

    def test_single_choice(self):
        row = _make_arc_row(choices=[{"label": "A", "text": "only"}])
        ex = normalize_arc_example(row, "validation", "ARC-Challenge")
        issues = ex.is_valid()
        assert any("2" in issue for issue in issues)

    def test_duplicate_labels(self):
        row = _make_arc_row(choices=[
            {"label": "A", "text": "x"},
            {"label": "A", "text": "y"},
        ])
        ex = normalize_arc_example(row, "validation", "ARC-Challenge")
        issues = ex.is_valid()
        assert any("duplicate" in issue.lower() for issue in issues)

    def test_empty_choice_text(self):
        row = _make_arc_row(choices=[
            {"label": "A", "text": ""},
            {"label": "B", "text": "ok"},
        ])
        ex = normalize_arc_example(row, "validation", "ARC-Challenge")
        issues = ex.is_valid()
        assert any("empty" in issue.lower() for issue in issues)


class TestPromptTemplate:
    def test_arc_v1_renders_correctly(self):
        ex = ARCExample(
            id="test",
            question="What color is the sky?",
            choices=[
                ARCChoice("A", "Blue"),
                ARCChoice("B", "Green"),
            ],
            answer_key="A",
            split="validation",
            config="ARC-Challenge",
        )
        tmpl = get_template("arc_v1")
        prompt = tmpl.render_prompt(ex)
        assert "What color is the sky?" in prompt
        assert "A. Blue" in prompt
        assert "B. Green" in prompt
        assert "Answer:" in prompt
        assert "The correct answer is" not in prompt

    def test_arc_label_v1_renders_label_prompt(self):
        ex = ARCExample(
            id="test",
            question="What color is the sky?",
            choices=[ARCChoice("C", "Red")],
            answer_key="C",
            split="validation",
            config="ARC-Challenge",
        )
        tmpl = get_template("arc_label_v1")
        prompt = tmpl.render_prompt(ex)
        assert "The correct answer is" in prompt

    def test_arc_v1_continuation(self):
        tmpl = get_template("arc_v1")
        cont = tmpl.render_continuation("some answer")
        assert cont == " some answer"

    def test_template_has_stable_hash(self):
        tmpl = get_template("arc_v1")
        h1 = tmpl.hash()
        h2 = tmpl.hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            get_template("nonexistent")


class TestPredictionScoring:
    def test_selects_highest_norm_score(self):
        scores = [
            ChoiceScore("A", "text a", -1.0, -5.0, 5),
            ChoiceScore("B", "text b", -0.5, -4.0, 8),
            ChoiceScore("C", "text c", -2.0, -10.0, 5),
        ]
        best = max(scores, key=lambda s: s.score_norm)
        assert best.label == "B"

    def test_selects_highest_sum_score(self):
        scores = [
            ChoiceScore("A", "short", -1.0, -5.0, 5),
            ChoiceScore("B", "much longer text", -2.0, -4.0, 2),
            ChoiceScore("C", "medium", -1.5, -3.0, 2),
        ]
        best = max(scores, key=lambda s: s.score_sum)
        assert best.label == "C"


class TestSummaryAccuracy:
    def test_accuracy_computation(self):
        scored = [
            ScoredExample(
                example=ARCExample(
                    id=str(i), question="q", choices=[
                        ARCChoice("A", "a"), ARCChoice("B", "b")
                    ],
                    answer_key="A", split="val", config="ARC-Challenge",
                ),
                scores=[], pred_norm="A", pred_sum="A",
                correct_norm=(i % 2 == 0), correct_sum=False, margin_norm=0.5, elapsed_sec=0.1,
            )
            for i in range(10)
        ]
        correct = sum(1 for s in scored if s.correct_norm)
        assert correct == 5
        assert correct / len(scored) == 0.5


class TestJsonlReportRoundtrip:
    def test_roundtrip(self):
        scored = [
            ScoredExample(
                example=ARCExample(
                    id="test_roundtrip", question="q?",
                    choices=[ARCChoice("A", "a"), ARCChoice("B", "b")],
                    answer_key="A", split="val", config="ARC-Challenge",
                ),
                scores=[
                    ChoiceScore("A", "a", -0.5, -1.0, 2),
                    ChoiceScore("B", "b", -1.5, -3.0, 2),
                ],
                pred_norm="A", pred_sum="A",
                correct_norm=True, correct_sum=True,
                margin_norm=1.0, elapsed_sec=0.05,
            )
        ]

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            meta = {
                "config": "ARC-Challenge",
                "dataset": "allenai/ai2_arc",
                "split": "validation",
                "model": "test-model",
                "mode": "hf-causal",
                "prompt_template": "arc_v1",
                "prompt_template_sha256": "abc123",
                "duration_sec": 1.0,
                "started_at": "",
            }
            write_reports(out, scored, 0, meta)

            assert (out / "summary.json").exists()
            assert (out / "predictions.jsonl").exists()
            assert (out / "failures.jsonl").exists()
            assert (out / "environment.json").exists()

            summary = json.loads((out / "summary.json").read_text())
            assert summary["accuracy_norm"] == 1.0
            assert summary["num_examples_scored"] == 1

            preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().strip().split("\n") if l]
            assert len(preds) == 1
            assert preds[0]["id"] == "test_roundtrip"
            assert preds[0]["correct_norm"] is True
            assert len(preds[0]["scores"]) == 2

            fails = [json.loads(l) for l in (out / "failures.jsonl").read_text().strip().split("\n") if l]
            assert len(fails) == 0

    def test_failures_captured(self):
        scored = [
            ScoredExample(
                example=ARCExample(
                    id="fail_1", question="q?",
                    choices=[ARCChoice("A", "a"), ARCChoice("B", "b")],
                    answer_key="A", split="val", config="ARC-Challenge",
                ),
                scores=[
                    ChoiceScore("A", "a", -1.0, -2.0, 2),
                    ChoiceScore("B", "b", -0.5, -1.0, 2),
                ],
                pred_norm="B", pred_sum="B",
                correct_norm=False, correct_sum=False,
                margin_norm=0.5, elapsed_sec=0.1,
            )
        ]

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            meta = {
                "config": "ARC-Challenge",
                "dataset": "test",
                "split": "val",
                "model": "test",
                "mode": "hf-causal",
                "prompt_template": "arc_v1",
                "prompt_template_sha256": "abc",
                "duration_sec": 1.0,
                "started_at": "",
            }
            write_reports(out, scored, 0, meta)

            summary = json.loads((out / "summary.json").read_text())
            assert summary["accuracy_norm"] == 0.0

            fails = [json.loads(l) for l in (out / "failures.jsonl").read_text().strip().split("\n") if l]
            assert len(fails) == 1
            assert fails[0]["id"] == "fail_1"


class TestNoGenerationRequired:
    def test_scorer_uses_logprobs_not_generate(self):
        from hybrid.benchmarks.arc_scoring import HFArcScorer
        assert hasattr(HFArcScorer, "score_options")
        assert not hasattr(HFArcScorer, "generate")


class TestDryRunWithJsonl:
    def test_load_from_local_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            jsonl_path = Path(td) / "test_arc.jsonl"
            row = _make_arc_row()
            jsonl_path.write_text(json.dumps(row) + "\n")

            valid, invalid, _ = load_arc_dataset(
                local_jsonl=str(jsonl_path),
                split="validation",
                config="ARC-Challenge",
            )
            assert len(valid) == 1
            assert len(invalid) == 0
            assert valid[0].id == "test_001"
            assert valid[0].answer_key == "A"

    def test_invalid_skipped_not_strict(self):
        with tempfile.TemporaryDirectory() as td:
            jsonl_path = Path(td) / "test_arc.jsonl"
            row = _make_arc_row(answer_key="Z")
            jsonl_path.write_text(json.dumps(row) + "\n")

            valid, invalid, _ = load_arc_dataset(
                local_jsonl=str(jsonl_path),
                split="validation",
                config="ARC-Challenge",
                strict_data=False,
            )
            assert len(valid) == 0
            assert len(invalid) == 1

    def test_invalid_raises_strict(self):
        with tempfile.TemporaryDirectory() as td:
            jsonl_path = Path(td) / "test_arc.jsonl"
            row = _make_arc_row(answer_key="Z")
            jsonl_path.write_text(json.dumps(row) + "\n")

            with pytest.raises(ValueError):
                load_arc_dataset(
                    local_jsonl=str(jsonl_path),
                    split="validation",
                    config="ARC-Challenge",
                    strict_data=True,
                )
