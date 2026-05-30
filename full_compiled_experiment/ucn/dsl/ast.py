from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActivateType(Enum):
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"
    IDENTITY = "identity"


class TypeSpec(Enum):
    VECTOR = "Vector"
    SUBSPACE = "Subspace"
    SCALAR = "Scalar"
    MATRIX = "Matrix"


class MatrixRef:
    def __init__(self, ref_type: str, name: str):
        self.ref_type = ref_type
        self.name = name

    def __repr__(self):
        return f"MatrixRef({self.ref_type}.{self.name})"


class SubspaceRef:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    @property
    def size(self) -> int:
        return self.end - self.start

    def __repr__(self):
        return f"SubspaceRef({self.start}:{self.end})"


class DBSpec:
    def __init__(self, partition: str):
        self.partition = partition

    def __repr__(self):
        return f"DBSpec(db.{self.partition})"


class ScalarExpr:
    def __init__(self, value: float):
        self.value = value

    def __repr__(self):
        return str(self.value)


class Expr:
    pass


@dataclass
class Mix(Expr):
    inputs: list[str]
    weights: list[float]

    def __post_init__(self):
        if len(self.inputs) != len(self.weights):
            raise ValueError(
                f"mix: {len(self.inputs)} inputs but {len(self.weights)} weights"
            )

    def __repr__(self):
        pairs = ", ".join(
            f"{v}@{w:.3f}" for v, w in zip(self.inputs, self.weights)
        )
        return f"mix([{pairs}])"


@dataclass
class Project(Expr):
    input: str
    subspace: SubspaceRef


@dataclass
class Transform(Expr):
    input: str
    matrix: MatrixRef


@dataclass
class Activate(Expr):
    input: str
    activation: ActivateType


@dataclass
class QueryMemory(Expr):
    input: str
    db: DBSpec
    top_k: int


@dataclass
class Residual(Expr):
    inputs: list[str]


@dataclass
class GatherContext(Expr):
    query: str
    source: str
    top_k: int = 0
    causal: bool = True


@dataclass
class Rotate(Expr):
    input: str
    theta: float
    subspace: SubspaceRef


@dataclass
class AllocDecl:
    name: str
    type_spec: TypeSpec
    dim: int = 0

    def __repr__(self):
        return f"alloc({self.name}, {self.type_spec.value}[{self.dim}])"


@dataclass
class Statement:
    target: str
    expr: Expr

    def __repr__(self):
        return f"{self.target} = {self.expr}"


@dataclass
class Program:
    declarations: list[AllocDecl] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)

    def add_decl(self, name: str, type_spec: TypeSpec, dim: int = 0):
        self.declarations.append(AllocDecl(name, type_spec, dim))

    def add_stmt(self, target: str, expr: Expr):
        self.statements.append(Statement(target, expr))

    def __repr__(self):
        parts = [repr(d) for d in self.declarations]
        parts += [repr(s) for s in self.statements]
        return ";\n".join(parts)


ASTNode = Program
