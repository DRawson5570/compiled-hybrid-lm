"""ARC prompt templates."""
from __future__ import annotations

import hashlib
from typing import Protocol

from hybrid.benchmarks.arc_data import ARCExample


class PromptTemplate(Protocol):
    template_id: str

    def render_prompt(self, example: ARCExample) -> str: ...
    def render_continuation(self, choice_text: str) -> str: ...
    def hash(self) -> str: ...


class _ArcV1Template:
    template_id = "arc_v1"

    def render_prompt(self, example: ARCExample) -> str:
        choices_block = "\n".join(
            f"{c.label}. {c.text}" for c in example.choices
        )
        return f"Question: {example.question}\n\nChoices:\n{choices_block}\n\nAnswer:"

    def render_continuation(self, choice_text: str) -> str:
        return f" {choice_text}"

    def hash(self) -> str:
        source = (
            'Question: {question}\n\nChoices:\n'
            '{choices}\n\nAnswer:'
        )
        return hashlib.sha256(source.encode()).hexdigest()


class _ArcLabelV1Template:
    template_id = "arc_label_v1"

    def render_prompt(self, example: ARCExample) -> str:
        choices_block = "\n".join(
            f"{c.label}. {c.text}" for c in example.choices
        )
        return (
            f"Question: {example.question}\n\nChoices:\n{choices_block}\n\n"
            f"The correct answer is"
        )

    def render_continuation(self, label: str) -> str:
        return f" {label}"

    def hash(self) -> str:
        source = (
            'Question: {question}\n\nChoices:\n'
            '{choices}\n\nThe correct answer is'
        )
        return hashlib.sha256(source.encode()).hexdigest()


_registry: dict[str, PromptTemplate] = {
    "arc_v1": _ArcV1Template(),
    "arc_label_v1": _ArcLabelV1Template(),
}


def get_template(template_id: str) -> PromptTemplate:
    if template_id not in _registry:
        raise ValueError(
            f"unknown prompt template {template_id!r}. "
            f"Available: {sorted(_registry.keys())}"
        )
    return _registry[template_id]
