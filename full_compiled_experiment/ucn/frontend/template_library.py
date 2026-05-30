from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

from ..dsl.ast import (
    Activate,
    ActivateType,
    DBSpec,
    MatrixRef,
    Mix,
    Program,
    Project,
    QueryMemory,
    Residual,
    Rotate,
    Statement,
    SubspaceRef,
    Transform,
)


@dataclass
class TemplateDef:
    template_id: int
    name: str
    description: str
    n_params: int
    fixed_expr: Optional[Any] = None


DEFAULT_TEMPLATES: List[TemplateDef] = [
    TemplateDef(
        template_id=0,
        name="identity_pass",
        description="Pass through with optional scaling",
        n_params=1,
    ),
    TemplateDef(
        template_id=1,
        name="single_transform",
        description="Apply a single stdlib transform",
        n_params=1,
    ),
    TemplateDef(
        template_id=2,
        name="mix_two",
        description="Weighted mix of two sources",
        n_params=1,
    ),
    TemplateDef(
        template_id=3,
        name="transform_activate",
        description="Transform then activate",
        n_params=2,
    ),
    TemplateDef(
        template_id=4,
        name="mix_activate",
        description="Mix two sources then activate",
        n_params=2,
    ),
    TemplateDef(
        template_id=5,
        name="rotate_transform",
        description="Rotate in subspace then transform",
        n_params=3,
    ),
    TemplateDef(
        template_id=6,
        name="project_transform",
        description="Project to subspace then transform",
        n_params=3,
    ),
    TemplateDef(
        template_id=7,
        name="dense_residual",
        description="Sum of multiple sources",
        n_params=0,
    ),
    TemplateDef(
        template_id=8,
        name="query_memory_lookup",
        description="Sparse key-value memory lookup",
        n_params=2,
    ),
]


class TemplateLibrary:
    def __init__(self, templates: List[TemplateDef] | None = None):
        self.templates = templates or DEFAULT_TEMPLATES
        self._id_map = {t.template_id: t for t in self.templates}

    def get(self, template_id: int) -> TemplateDef:
        return self._id_map[template_id]

    @property
    def n_templates(self) -> int:
        return len(self.templates)

    def build_program(
        self,
        template_id: int,
        params: List[float],
        stdlib_names: List[str] | None = None,
        input_name: str = "x",
        output_name: str = "y",
    ) -> Program:
        program = Program()

        if stdlib_names is None:
            stdlib_names = [f"primitive_{i}" for i in range(10)]

        n_lib = len(stdlib_names)
        n_acts = 4

        if template_id == 0:
            scale = params[0] if params else 1.0
            program.add_stmt("y", Mix([input_name], [scale]))

        elif template_id == 1:
            p = params[0] if params else 0.0
            matrix_idx = max(0, min(int(p * n_lib), n_lib - 1))
            matrix_ref = MatrixRef("stdlib", stdlib_names[matrix_idx])
            program.add_stmt("y", Transform(input_name, matrix_ref))

        elif template_id == 2:
            weight = params[0] if params else 0.5
            weight = max(0.0, min(1.0, weight))
            program.add_stmt("y", Mix([input_name, "prev_x"], [weight, 1.0 - weight]))

        elif template_id == 3:
            matrix_idx = max(0, min(int(params[0] * n_lib) if params else 0, n_lib - 1))
            act_idx = max(0, min(int(params[1] * n_acts) if len(params) > 1 else 0, n_acts - 1))
            acts = [ActivateType.IDENTITY, ActivateType.RELU, ActivateType.GELU, ActivateType.SILU]
            matrix_ref = MatrixRef("stdlib", stdlib_names[matrix_idx])
            program.add_stmt("t1", Transform(input_name, matrix_ref))
            program.add_stmt("y", Activate("t1", acts[act_idx]))

        elif template_id == 4:
            weight = params[0] if params else 0.5
            weight = max(0.0, min(1.0, weight))
            act_idx = max(0, min(int(params[1] * n_acts) if len(params) > 1 else 0, n_acts - 1))
            acts = [ActivateType.IDENTITY, ActivateType.RELU, ActivateType.GELU, ActivateType.SILU]
            program.add_stmt("t1", Mix([input_name, "prev_x"], [weight, 1.0 - weight]))
            program.add_stmt("y", Activate("t1", acts[act_idx]))

        elif template_id == 5:
            theta = params[0] * 6.2832 if params else 0.0
            p1 = params[1] if len(params) > 1 else 0.0
            matrix_idx = max(0, min(int(p1 * n_lib), n_lib - 1))
            subspace_end = max(4, int(params[2] * 1536) if len(params) > 2 else 64)
            matrix_ref = MatrixRef("stdlib", stdlib_names[matrix_idx])
            program.add_stmt("t1", Rotate(input_name, theta, SubspaceRef(0, subspace_end)))
            program.add_stmt("y", Transform("t1", matrix_ref))

        elif template_id == 6:
            p0 = params[0] if params else 0.0
            p1 = params[1] if len(params) > 1 else 0.5
            p2 = params[2] if len(params) > 2 else 0.0
            ss_start = max(0, int(p0 * 1536))
            ss_end = max(ss_start + 4, int(p1 * 1536))
            matrix_idx = max(0, min(int(p2 * n_lib), n_lib - 1))
            matrix_ref = MatrixRef("stdlib", stdlib_names[matrix_idx])
            program.add_stmt("t1", Project(input_name, SubspaceRef(ss_start, ss_end)))
            program.add_stmt("y", Transform("t1", matrix_ref))

        elif template_id == 7:
            if len(params) > 0:
                pass
            program.add_stmt("y", Residual([input_name]))

        elif template_id == 8:
            p0 = params[0] if params else 0.0
            p1 = params[1] if len(params) > 1 else 0.0
            top_k = max(1, int(p0 * 64) + 1)
            db_idx = max(0, min(int(p1 * len(stdlib_names)), len(stdlib_names) - 1))
            db_name = stdlib_names[db_idx]
            program.add_stmt("y", QueryMemory(input_name, DBSpec(db_name), top_k))

        else:
            program.add_stmt("y", Mix([input_name], [1.0]))

        return program

    def build_program_soft(
        self,
        template_weights: torch.Tensor,
        params: torch.Tensor,
        stdlib_names: List[str] | None = None,
        input_name: str = "x",
        output_name: str = "y",
    ) -> Program:
        """
        Differentiable program synthesis: uses soft weights over templates
        and stdlib entries instead of discrete argmax selection.
        
        template_weights: [n_templates] softmax distribution
        params: [n_templates, max_params] per-template parameters
        """
        import torch
        from ..dsl.ast import MatrixRef

        if stdlib_names is None:
            stdlib_names = [f"primitive_{i}" for i in range(10)]

        program = Program()
        n_templates = template_weights.shape[0]
        n_lib = len(stdlib_names)

        accumulator = MixBase(input_name, [1.0])

        for tid in range(n_templates):
            w = template_weights[tid]
            if w < 0.01:
                continue

            p = params[tid] if tid < params.shape[0] else torch.zeros(params.shape[1])
            tname = f"t{tid}"

            if tid == 0:
                scale = float(p[0].item()) if p.numel() > 0 else 1.0
                program.add_stmt(tname, Mix([input_name], [scale]))

            elif tid == 1:
                lib_idx = max(0, min(int(float(p[0].item()) * n_lib) if p.numel() > 0 else 0, n_lib - 1))
                program.add_stmt(tname, Transform(input_name, MatrixRef("stdlib", stdlib_names[lib_idx])))

            elif tid == 2:
                weight = max(0.0, min(1.0, float(p[0].item()))) if p.numel() > 0 else 0.5
                program.add_stmt(tname, Mix([input_name, "prev_x"], [weight, 1.0 - weight]))

            elif tid == 3:
                lib_idx = max(0, min(int(float(p[0].item()) * n_lib) if p.numel() > 0 else 0, n_lib - 1))
                act_idx = max(0, min(int(float(p[1].item()) * 4) if p.numel() > 1 else 0, 3))
                program.add_stmt(tname + "_t", Transform(input_name, MatrixRef("stdlib", stdlib_names[lib_idx])))
                program.add_stmt(tname, Activate(tname + "_t", [ActivateType.IDENTITY, ActivateType.RELU, ActivateType.GELU, ActivateType.SILU][act_idx]))

            elif tid >= 4:
                lib_idx = max(0, min(int(float(p[0].item()) * n_lib) if p.numel() > 0 else 0, n_lib - 1))
                program.add_stmt(tname, Transform(input_name, MatrixRef("stdlib", stdlib_names[lib_idx])))

            if tid > 0 and w > 0.0:
                accumulator.inputs.append(tname)
                accumulator.weights.append(float(w.item()))

        if len(accumulator.inputs) > 1:
            program.add_stmt(output_name, Mix(accumulator.inputs, accumulator.weights))
        elif len(accumulator.inputs) == 1:
            program.add_stmt(output_name, Mix([accumulator.inputs[0]], [accumulator.weights[0]]))
        else:
            program.add_stmt(output_name, Mix([input_name], [1.0]))

        return program


class MixBase:
    def __init__(self, input_name, weights):
        self.inputs = [input_name]
        self.weights = list(weights)
