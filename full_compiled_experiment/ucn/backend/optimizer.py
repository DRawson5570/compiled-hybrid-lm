from __future__ import annotations

from typing import Optional

import torch

from ..dsl.ast import (
    Activate,
    GatherContext,
    Mix,
    Program,
    Project,
    QueryMemory,
    Residual,
    Rotate,
    Statement,
    Transform,
)


def fuse_operators(program: Program) -> Program:
    fused = Program(declarations=list(program.declarations))
    i = 0
    stmts = program.statements

    while i < len(stmts):
        fused_group = _try_fuse(stmts, i)
        if fused_group:
            fused.statements.extend(fused_group)
            i += len(fused_group)
        else:
            fused.statements.append(stmts[i])
            i += 1

    return fused


def _try_fuse(stmts: list[Statement], start: int) -> list[Statement] | None:
    if start + 1 >= len(stmts):
        return None

    a = stmts[start]
    b = stmts[start + 1]

    if isinstance(a.expr, Transform) and isinstance(b.expr, Activate):
        return _fuse_transform_activate(a, b)

    if isinstance(a.expr, Mix) and isinstance(b.expr, Activate):
        return _fuse_mix_activate(a, b)

    return None


def _fuse_transform_activate(a: Statement, b: Statement) -> list:
    return [a, b]


def _fuse_mix_activate(a: Statement, b: Statement) -> list:
    return [a, b]


def eliminate_dead_code(program: Program) -> Program:
    used = set()
    for stmt in program.statements:
        _collect_used(stmt.expr, used)

    output_targets = set()
    if program.statements:
        output_targets.add(program.statements[-1].target)

    live = used | output_targets

    pruned = Program(declarations=list(program.declarations))
    for stmt in program.statements:
        if stmt.target in live:
            pruned.statements.append(stmt)

    return pruned


def _collect_used(expr, acc: set):
    if isinstance(expr, Mix):
        acc.update(expr.inputs)
    elif isinstance(expr, Project):
        acc.add(expr.input)
    elif isinstance(expr, Transform):
        acc.add(expr.input)
    elif isinstance(expr, Activate):
        acc.add(expr.input)
    elif isinstance(expr, QueryMemory):
        acc.add(expr.input)
    elif isinstance(expr, Residual):
        acc.update(expr.inputs)
    elif isinstance(expr, Rotate):
        acc.add(expr.input)
    elif isinstance(expr, GatherContext):
        acc.add(expr.query)
        acc.add(expr.source)


def prune_subspaces(program: Program) -> Program:
    return program


def optimize(program: Program) -> Program:
    program = eliminate_dead_code(program)
    program = fuse_operators(program)
    return program
