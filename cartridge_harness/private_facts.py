"""Synthetic private-fact task suite for cartridge knowledge injection."""
from __future__ import annotations

from hybrid.cartridge_harness.core import TaskExample


PRIVATE_FACTS: tuple[tuple[str, str], ...] = (
    ("Project Aster", "RAVEN-041"),
    ("Project Brindle", "MICA-772"),
    ("Project Cobalt", "ORBIT-318"),
    ("Project Drift", "LUMEN-506"),
    ("Project Ember", "NOVA-229"),
    ("Project Fjord", "QUARTZ-184"),
    ("Project Glade", "ION-693"),
    ("Project Helix", "VANTA-057"),
    ("Project Iris", "SOLACE-440"),
    ("Project Juno", "PULSE-815"),
    ("Project Kestrel", "AXIS-362"),
    ("Project Lattice", "CIPHER-904"),
)

TRAIN_TEMPLATES: tuple[str, ...] = (
    "Aurora private registry. Answer with only the registry value. What is the access code for {name}?\nAnswer:",
    "Internal Aurora lookup. Return only the code. {name} is assigned which registry code?\nAnswer:",
    "Use the sealed Aurora facts. Give just the value. Registry code for {name}:\nAnswer:",
)

HELDOUT_TEMPLATES: tuple[str, ...] = (
    "According to the Aurora internal registry, which access value unlocks {name}? Answer with only the code.\nAnswer:",
    "For {name}, report the private Aurora registry code and nothing else.\nAnswer:",
)


def build_private_fact_tasks(
    train_templates: tuple[str, ...] = TRAIN_TEMPLATES,
    heldout_templates: tuple[str, ...] = HELDOUT_TEMPLATES,
    facts: tuple[tuple[str, str], ...] = PRIVATE_FACTS,
) -> list[TaskExample]:
    """Build train and held-out paraphrase tasks from synthetic facts."""

    tasks: list[TaskExample] = []
    for split, templates in (("train", train_templates), ("heldout", heldout_templates)):
        for fact_idx, (name, answer) in enumerate(facts):
            for template_idx, template in enumerate(templates):
                tasks.append(
                    TaskExample(
                        task_id=f"{split}_{fact_idx}_{template_idx}",
                        split=split,
                        prompt=template.format(name=name),
                        expected=answer,
                        metadata={"name": name},
                    )
                )
    return tasks