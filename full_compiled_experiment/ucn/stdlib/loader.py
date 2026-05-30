from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch

from .schema import BehaviorMeta, MathDef, PrimitiveEntry


def load_stdlib(path: str | Path) -> Dict[str, PrimitiveEntry]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"stdlib not found: {path}")

    with open(path) as f:
        data = json.load(f)

    entries = {}
    for pid, entry_data in data["primitives"].items():
        math_def = MathDef(**entry_data["mathematical_definition"])
        behavior = BehaviorMeta(**entry_data["behavioral_metadata"])
        entry = PrimitiveEntry(
            primitive_id=entry_data["primitive_id"],
            symbolic_name=entry_data["symbolic_name"],
            type=entry_data["type"],
            source_layers=entry_data["source_layers"],
            math_def=math_def,
            behavior=behavior,
            weight_data=entry_data.get("weight_data", {}),
        )
        entries[pid] = entry
    return entries


def resolve_weights(
    entry: PrimitiveEntry,
    weights_dir: str | Path,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    weights_dir = Path(weights_dir)
    weights = {}

    if entry.math_def.u_uri:
        u_path = weights_dir / Path(entry.math_def.u_uri).name
        if u_path.exists():
            weights["u"] = torch.load(u_path, map_location=device, weights_only=True)

    if entry.math_def.v_uri:
        v_path = weights_dir / Path(entry.math_def.v_uri).name
        if v_path.exists():
            weights["v"] = torch.load(v_path, map_location=device, weights_only=True)

    if entry.math_def.vector_uri:
        vec_path = weights_dir / Path(entry.math_def.vector_uri).name
        if vec_path.exists():
            weights["vector"] = torch.load(vec_path, map_location=device, weights_only=True)

    return weights


def save_stdlib_json(
    entries: list[PrimitiveEntry],
    path: str | Path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    primitives = {}
    for entry in entries:
        primitives[entry.primitive_id] = {
            "primitive_id": entry.primitive_id,
            "symbolic_name": entry.symbolic_name,
            "type": entry.type,
            "source_layers": entry.source_layers,
            "mathematical_definition": {
                "operator_type": entry.math_def.operator_type,
                "rank": entry.math_def.rank,
                "u_uri": entry.math_def.u_uri,
                "v_uri": entry.math_def.v_uri,
                "vector_uri": entry.math_def.vector_uri,
            },
            "behavioral_metadata": {
                "description": entry.behavior.description,
                "trigger_conditions": entry.behavior.trigger_conditions,
            },
            "weight_data": entry.weight_data,
        }

    with open(path, "w") as f:
        json.dump({"stdlib_version": "3.0.0", "primitives": primitives}, f, indent=2)


def save_weight_tensor(tensor: torch.Tensor, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)
