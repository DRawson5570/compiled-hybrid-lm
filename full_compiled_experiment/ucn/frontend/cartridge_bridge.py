from __future__ import annotations

from ..dsl.ast import Program


_TEMPLATE_TO_CARTRIDGE = {
    0: None,
    1: "qwen-arc-challenge-cartridge",
    2: None,
    3: "qwen-arithmetic-router-cartridge",
    4: "qwen-code-router-cartridge",
    5: "qwen-safety-router-cartridge",
    6: "qwen-instruction-format-cartridge",
    7: None,
    8: "qwen-hellaswag-cartridge",
}


def program_to_cartridge_id(
    program: Program,
    template_map: dict | None = None,
) -> str | None:
    if template_map is None:
        template_map = _TEMPLATE_TO_CARTRIDGE

    if not program.statements:
        return None

    if hasattr(program, "template_id"):
        template_id = program.template_id
    else:
        for stmt in program.statements:
            from ..dsl.ast import Transform, QueryMemory, Mix, Activate

            expr = stmt.expr
            if isinstance(expr, Transform) and expr.matrix.ref_type == "stdlib":
                name = expr.matrix.name.lower()
                for cid in [
                    "private-facts", "arithmetic", "code", "safety",
                    "instruction-format", "arc-challenge",
                ]:
                    if cid in name:
                        return f"qwen-{cid}-cartridge"

            if isinstance(expr, QueryMemory):
                partition = expr.db.partition.lower()
                for cid in ["private", "arithmetic", "code", "safety"]:
                    if cid in partition:
                        return f"qwen-{cid}-router-cartridge"

        template_id = 0
        for stmt in program.statements:
            if isinstance(stmt.expr, Mix) and len(stmt.expr.inputs) == 1:
                template_id = 0
            elif isinstance(stmt.expr, Transform):
                template_id = 1
            elif isinstance(stmt.expr, (Mix,)) and len(stmt.expr.inputs) > 1:
                template_id = 2
            elif isinstance(stmt.expr, Activate):
                template_id = 3

    return template_map.get(template_id)
