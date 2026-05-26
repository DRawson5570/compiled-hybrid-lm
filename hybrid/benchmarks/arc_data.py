"""ARC benchmark dataset normalization and loading."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ARCChoice:
    label: str
    text: str


@dataclass
class ARCExample:
    id: str
    question: str
    choices: list[ARCChoice]
    answer_key: str | None
    split: str
    config: str

    def is_valid(self) -> list[str]:
        issues: list[str] = []
        if not self.question.strip():
            issues.append("empty question")
        if len(self.choices) < 2:
            issues.append(f"only {len(self.choices)} choices (need >= 2)")
        labels = [c.label for c in self.choices]
        if len(labels) != len(set(labels)):
            issues.append("duplicate choice labels")
        for c in self.choices:
            if not c.text.strip():
                issues.append(f"empty choice text for label {c.label!r}")
        if self.answer_key is not None and self.answer_key not in labels:
            issues.append(f"answer_key {self.answer_key!r} not in choices {labels}")
        return issues


def _normalize_choices(choices_raw) -> list[ARCChoice]:
    if isinstance(choices_raw, dict):
        labels = choices_raw.get("label", [])
        texts = choices_raw.get("text", [])
        if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
            return [ARCChoice(label=str(l), text=str(t)) for l, t in zip(labels, texts)]
    if isinstance(choices_raw, list):
        return [ARCChoice(label=str(c["label"]), text=str(c["text"])) for c in choices_raw]
    return []


def normalize_arc_example(raw: dict, split: str, config: str) -> ARCExample:
    choices = _normalize_choices(raw.get("choices", raw.get("answer_choices", [])))
    answer_key = raw.get("answerKey") or raw.get("answer_key") or None
    if isinstance(answer_key, (int, float)):
        answer_key = str(answer_key)
    return ARCExample(
        id=raw["id"],
        question=raw["question"],
        choices=choices,
        answer_key=answer_key,
        split=split,
        config=config,
    )


def load_arc_dataset(
    dataset_name: str = "allenai/ai2_arc",
    config: str = "ARC-Challenge",
    split: str = "validation",
    local_jsonl: str | None = None,
    max_examples: int = 0,
    strict_data: bool = False,
) -> tuple[list[ARCExample], list[ARCExample], list[tuple[dict, list[str]]]]:
    if local_jsonl:
        with open(local_jsonl, encoding="utf-8") as fh:
            raw_examples = [json.loads(line) for line in fh if line.strip()]
    else:
        from datasets import load_dataset
        try:
            ds = load_dataset(dataset_name, config, trust_remote_code=True)
        except (TypeError, ValueError):
            ds = load_dataset(dataset_name, config)
        raw_examples = [dict(item) for item in ds[split]]

    valid: list[ARCExample] = []
    invalid: list[ARCExample] = []
    invalid_raw: list[tuple[dict, list[str]]] = []

    for raw in raw_examples:
        example = normalize_arc_example(raw, split, config)
        issues = example.is_valid()
        if issues:
            invalid.append(example)
            invalid_raw.append((raw, issues))
            if strict_data:
                raise ValueError(
                    f"invalid example {example.id!r}: {'; '.join(issues)}\n"
                    f"raw: {json.dumps(raw)}"
                )
            continue
        valid.append(example)

    if max_examples > 0:
        valid = valid[:max_examples]

    return valid, invalid, invalid_raw
