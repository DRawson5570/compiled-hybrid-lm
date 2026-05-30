from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MathDef:
    operator_type: str
    rank: Optional[int] = None
    u_uri: Optional[str] = None
    v_uri: Optional[str] = None
    vector_uri: Optional[str] = None


@dataclass
class BehaviorMeta:
    description: str = ""
    trigger_conditions: list[str] = field(default_factory=list)


@dataclass
class PrimitiveEntry:
    primitive_id: str
    symbolic_name: str
    type: str
    source_layers: list[int]
    math_def: MathDef
    behavior: BehaviorMeta
    weight_data: dict = field(default_factory=dict)
