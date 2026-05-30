from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import triton
import triton.language as tl

from ...dsl.ast import (
    Activate,
    ActivateType,
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


@triton.jit
def _mix_kernel(
    x_ptrs_ptr,
    y_ptr,
    weights_ptr,
    n_vecs: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(n_vecs):
        w = tl.load(weights_ptr + i)
        x_ptr = tl.load(x_ptrs_ptr + i).to(tl.pointer_type(tl.float32))
        val = tl.load(x_ptr + offsets, mask=mask)
        accum += val.to(tl.float32) * w

    tl.store(y_ptr + offsets, accum, mask=mask)


@triton.jit
def _project_kernel(
    x_ptr,
    y_ptr,
    start_idx: tl.constexpr,
    end_idx: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(x_ptr + offsets, mask=mask)
    is_in_range = (offsets >= start_idx) & (offsets < end_idx)
    result = tl.where(is_in_range, val, 0.0)
    tl.store(y_ptr + offsets, result, mask=mask)


@triton.jit
def _transform_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    d_in: tl.constexpr,
    d_out: tl.constexpr,
    BLOCK_SIZE_OUT: tl.constexpr,
    BLOCK_SIZE_IN: tl.constexpr,
):
    pid = tl.program_id(0)
    out_offsets = pid * BLOCK_SIZE_OUT + tl.arange(0, BLOCK_SIZE_OUT)
    out_mask = out_offsets < d_out

    accum = tl.zeros((BLOCK_SIZE_OUT,), dtype=tl.float32)
    for i_start in range(0, d_in, BLOCK_SIZE_IN):
        in_offsets = i_start + tl.arange(0, BLOCK_SIZE_IN)
        in_mask = in_offsets < d_in
        x_vals = tl.load(x_ptr + in_offsets, mask=in_mask, other=0.0)
        for j in range(BLOCK_SIZE_OUT):
            if out_offsets[j] < d_out:
                w_offsets = out_offsets[j] * d_in + in_offsets
                w_vals = tl.load(w_ptr + w_offsets, mask=in_mask, other=0.0)
                accum_j = tl.sum(x_vals.to(tl.float32) * w_vals.to(tl.float32))
                accum = accum.to(tl.float32)
                idx = tl.arange(0, BLOCK_SIZE_OUT)
                accum = tl.where(idx == j, accum + accum_j.to(tl.float32), accum)

    tl.store(y_ptr + out_offsets, accum, mask=out_mask)


@triton.jit
def _activate_kernel(
    x_ptr,
    y_ptr,
    n_elements: tl.constexpr,
    act_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    if act_type == 0:
        result = val
    elif act_type == 1:
        result = tl.maximum(val, 0.0)
    elif act_type == 2:
        inv_sqrt2 = 0.7071067811865476
        result = 0.5 * val * (1.0 + tl.math.erf(val * inv_sqrt2))
    elif act_type == 3:
        result = val * tl.sigmoid(val)
    else:
        result = val

    tl.store(y_ptr + offsets, result, mask=mask)


@triton.jit
def _fused_transform_activate_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    d_in: tl.constexpr,
    d_out: tl.constexpr,
    act_type: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    out_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offsets < d_out

    accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i_start in range(0, d_in, TILE_SIZE):
        in_offsets = i_start + tl.arange(0, TILE_SIZE)
        in_mask = in_offsets < d_in
        x_vals = tl.load(x_ptr + in_offsets, mask=in_mask, other=0.0).to(tl.float32)

        w_base = (out_offsets[:, None] * d_in + in_offsets[None, :])
        w_mask = out_mask[:, None] & in_mask[None, :]
        w_vals = tl.load(w_ptr + w_base, mask=w_mask, other=0.0).to(tl.float32)

        accum += tl.sum(x_vals[None, :] * w_vals, axis=1)

    if act_type == 1:
        accum = tl.maximum(accum, 0.0)
    elif act_type == 2:
        inv_sqrt2 = 0.7071067811865476
        accum = 0.5 * accum * (1.0 + tl.math.erf(accum * inv_sqrt2))
    elif act_type == 3:
        accum = accum * tl.sigmoid(accum)

    tl.store(y_ptr + out_offsets, accum, mask=out_mask)


@triton.jit
def _residual_kernel(
    x_ptrs_ptr,
    y_ptr,
    n_inputs: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(n_inputs):
        x_ptr = tl.load(x_ptrs_ptr + i).to(tl.pointer_type(tl.float32))
        val = tl.load(x_ptr + offsets, mask=mask)
        accum += val.to(tl.float32)

    tl.store(y_ptr + offsets, accum, mask=mask)


@triton.jit
def _rotate_kernel(
    x_ptr,
    y_ptr,
    start_idx: tl.constexpr,
    end_idx: tl.constexpr,
    cos_theta: tl.constexpr,
    sin_theta: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    half = (end_idx - start_idx) // 2

    is_half0 = (offsets >= start_idx) & (offsets < start_idx + half)

    val_a = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    paired_offsets = offsets + half
    paired_mask = paired_offsets < n_elements
    val_b = tl.load(x_ptr + paired_offsets, mask=paired_mask).to(tl.float32)

    rotated_half0 = val_a * cos_theta - val_b * sin_theta
    rotated_half1 = val_a * sin_theta + val_b * cos_theta

    result = tl.where(is_half0, rotated_half0, rotated_half1)
    result = tl.where(mask, result, 0.0)

    tl.store(y_ptr + offsets, result, mask=mask)


@triton.jit
def _gather_context_kernel(
    q_ptr,
    src_ptr,
    y_ptr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    t_offsets = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T

    m_i = tl.full((BLOCK_T,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_T,), dtype=tl.float32)
    accum = tl.zeros((BLOCK_T, D), dtype=tl.float32)

    scale = D ** -0.5

    for s_start in range(0, T, BLOCK_T):
        s_offsets = s_start + tl.arange(0, BLOCK_T)
        s_mask = s_offsets < T

        s_filter = (tl.arange(0, BLOCK_T) < BLOCK_T)
        s_val = tl.load(src_ptr + (s_offsets[:, None] * D + tl.arange(0, D)[None, :]),
                        mask=s_mask[:, None] & (tl.arange(0, D) < D)[None, :]).to(tl.float32)
        s_val = tl.where(s_mask[:, None], s_val, 0.0)

        q_val = tl.load(q_ptr + (t_offsets[:, None] * D + tl.arange(0, D)[None, :]),
                        mask=t_mask[:, None] & (tl.arange(0, D) < D)[None, :]).to(tl.float32)
        q_val = tl.where(t_mask[:, None], q_val, 0.0)

        scores = tl.dot(q_val, tl.trans(s_val)) * scale

        causal = (t_offsets[:, None] >= s_offsets[None, :])
        scores = tl.where(causal & t_mask[:, None] & s_mask[None, :], scores, float("-inf"))

        m_prev = m_i
        m_curr = tl.max(scores, axis=1)
        m_i = tl.maximum(m_prev, m_curr)

        alpha = tl.exp(m_prev - m_i)
        p = tl.exp(scores - m_i[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)

        accum = accum * alpha[:, None] + tl.dot(p, s_val)

    accum = accum / l_i[:, None]
    tl.store(y_ptr + t_offsets[:, None] * D + tl.arange(0, D)[None, :],
             accum, mask=t_mask[:, None] & (tl.arange(0, D) < D)[None, :])


def _act_type_code(act: ActivateType) -> int:
    return {
        ActivateType.IDENTITY: 0,
        ActivateType.RELU: 1,
        ActivateType.GELU: 2,
        ActivateType.SILU: 3,
    }.get(act, 0)


class TritonBackend:
    def __init__(
        self,
        stdlib_weights: Dict[str, Any] | None = None,
        device: str = "cuda",
        block_size: int = 256,
    ):
        self.stdlib_weights = stdlib_weights or {}
        self.device = device
        self.block_size = block_size
        self._compiled_kernels: Dict[str, Callable] = {}

    def compile(self, program: Program) -> Callable:
        stmts = list(program.statements)
        if not stmts:
            return lambda inputs, batch_size=None: inputs

        fused_ops = self._fuse_consecutive(stmts)

        def triton_executor(
            inputs: Dict[str, torch.Tensor],
            batch_size: int | None = None,
        ) -> Dict[str, torch.Tensor]:
            ws = dict(inputs)
            for op in fused_ops:
                result = self._dispatch(op, ws, batch_size)
                ws[op["target"]] = result
            return ws

        return triton_executor

    def _fuse_consecutive(self, stmts: list[Statement]) -> list[Dict[str, Any]]:
        ops = []
        i = 0
        while i < len(stmts):
            a = stmts[i]
            if (
                i + 1 < len(stmts)
                and isinstance(a.expr, Transform)
                and isinstance(stmts[i + 1].expr, Activate)
            ):
                b = stmts[i + 1]
                ops.append({
                    "type": "fused_transform_activate",
                    "target": b.target,
                    "input": a.expr.input,
                    "matrix": a.expr.matrix,
                    "activation": b.expr.activation,
                })
                i += 2
            else:
                ops.append(self._stmt_to_op(a))
                i += 1
        return ops

    def _stmt_to_op(self, stmt: Statement) -> Dict[str, Any]:
        expr = stmt.expr
        if isinstance(expr, Mix):
            return {
                "type": "mix",
                "target": stmt.target,
                "inputs": expr.inputs,
                "weights": expr.weights,
            }
        elif isinstance(expr, Project):
            return {
                "type": "project",
                "target": stmt.target,
                "input": expr.input,
                "subspace": expr.subspace,
            }
        elif isinstance(expr, Transform):
            return {
                "type": "transform",
                "target": stmt.target,
                "input": expr.input,
                "matrix": expr.matrix,
            }
        elif isinstance(expr, Activate):
            return {
                "type": "activate",
                "target": stmt.target,
                "input": expr.input,
                "activation": expr.activation,
            }
        elif isinstance(expr, Residual):
            return {
                "type": "residual",
                "target": stmt.target,
                "inputs": expr.inputs,
            }
        elif isinstance(expr, Rotate):
            return {
                "type": "rotate",
                "target": stmt.target,
                "input": expr.input,
                "theta": expr.theta,
                "subspace": expr.subspace,
            }
        elif isinstance(expr, GatherContext):
            return {
                "type": "gather_context",
                "target": stmt.target,
                "query": expr.query,
                "source": expr.source,
                "top_k": expr.top_k,
                "causal": expr.causal,
            }
        elif isinstance(expr, QueryMemory):
            return {
                "type": "query_memory",
                "target": stmt.target,
                "input": expr.input,
                "db": expr.db.partition,
                "top_k": expr.top_k,
            }
        else:
            return {
                "type": "pass",
                "target": stmt.target,
                "input": expr.input if hasattr(expr, "input") else "input",
            }

    def _dispatch(
        self,
        op: Dict[str, Any],
        ws: Dict[str, torch.Tensor],
        batch_size: int | None,
    ) -> torch.Tensor:
        handlers = {
            "mix": self._run_mix,
            "project": self._run_project,
            "transform": self._run_transform,
            "activate": self._run_activate,
            "fused_transform_activate": self._run_fused_transform_activate,
            "residual": self._run_residual,
            "rotate": self._run_rotate,
            "gather_context": self._run_gather_context,
            "query_memory": self._run_query_memory,
        }
        handler = handlers.get(op["type"])
        if handler:
            return handler(op, ws, batch_size)
        elif op["type"] == "pass":
            input_name = op.get("input", "input")
            return ws.get(input_name, torch.zeros(1))
        else:
            raise ValueError(f"Unknown op type: {op['type']}")

    def _run_mix(self, op, ws, batch_size) -> torch.Tensor:
        inputs = [ws[name].contiguous() for name in op["inputs"]]
        n_elements = inputs[0].numel()

        x_ptrs = torch.tensor(
            [v.data_ptr() for v in inputs],
            dtype=torch.int64,
            device=self.device,
        )
        weights = torch.tensor(
            op["weights"],
            dtype=torch.float32,
            device=self.device,
        )
        output = torch.empty_like(inputs[0])

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _mix_kernel[grid](
            x_ptrs,
            output,
            weights,
            n_vecs=len(inputs),
            n_elements=n_elements,
            BLOCK_SIZE=self.block_size,
        )
        return output

    def _run_project(self, op, ws, batch_size) -> torch.Tensor:
        x = ws[op["input"]].contiguous()
        ss = op["subspace"]
        n_elements = x.numel()
        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _project_kernel[grid](
            x,
            output,
            start_idx=ss.start,
            end_idx=ss.end,
            n_elements=n_elements,
            BLOCK_SIZE=self.block_size,
        )
        return output

    def _run_transform(self, op, ws, batch_size) -> torch.Tensor:
        return self._apply_transform_gpu(op, ws)

    def _run_activate(self, op, ws, batch_size) -> torch.Tensor:
        x = ws[op["input"]].contiguous()
        n_elements = x.numel()
        act_code = _act_type_code(op["activation"])
        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _activate_kernel[grid](
            x,
            output,
            n_elements=n_elements,
            act_type=act_code,
            BLOCK_SIZE=self.block_size,
        )
        return output

    def _run_fused_transform_activate(self, op, ws, batch_size) -> torch.Tensor:
        x = ws[op["input"]].contiguous()
        d_in = x.shape[-1]
        act_code = _act_type_code(op["activation"])

        matrix_ref = op["matrix"]
        if matrix_ref.ref_type == "stdlib" and matrix_ref.name in self.stdlib_weights:
            w = self._get_transform_weight(matrix_ref.name, d_in)
            w = w.to(device=x.device, dtype=torch.float32).contiguous()
            d_out = w.shape[0] if hasattr(w, 'shape') and w.dim() >= 1 else d_in
        else:
            w = torch.eye(d_in, device=x.device, dtype=torch.float32)

        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(d_out, meta["BLOCK_SIZE"]),)
        _fused_transform_activate_kernel[grid](
            x,
            output,
            w,
            d_in=d_in,
            d_out=d_out,
            act_type=act_code,
            BLOCK_SIZE=self.block_size,
            TILE_SIZE=min(128, d_in),
        )
        return output

    def _run_residual(self, op, ws, batch_size) -> torch.Tensor:
        inputs = [ws[name].contiguous() for name in op["inputs"]]
        n_elements = inputs[0].numel()

        x_ptrs = torch.tensor(
            [v.data_ptr() for v in inputs],
            dtype=torch.int64,
            device=self.device,
        )
        output = torch.empty_like(inputs[0])

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _residual_kernel[grid](
            x_ptrs,
            output,
            n_inputs=len(inputs),
            n_elements=n_elements,
            BLOCK_SIZE=self.block_size,
        )
        return output

    def _run_rotate(self, op, ws, batch_size) -> torch.Tensor:
        x = ws[op["input"]].contiguous()
        ss = op["subspace"]
        theta = op["theta"]
        n_elements = x.numel()
        cos_t = float(torch.cos(torch.tensor(theta)))
        sin_t = float(torch.sin(torch.tensor(theta)))
        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _rotate_kernel[grid](
            x,
            output,
            start_idx=ss.start,
            end_idx=ss.end,
            cos_theta=cos_t,
            sin_theta=sin_t,
            n_elements=n_elements,
            BLOCK_SIZE=self.block_size,
        )
        return output

    def _run_gather_context(self, op, ws, batch_size) -> torch.Tensor:
        q = ws[op["query"]].contiguous()
        src = ws[op["source"]].contiguous()

        if q.dim() == 2:
            q = q.unsqueeze(0)
            sq = True
        else:
            sq = False
        if src.dim() == 2:
            src = src.unsqueeze(0)
            ss = True
        else:
            ss = False

        B, T, D = q.shape
        _, Ts, _ = src.shape

        if T != Ts or B > 1 or op.get("top_k", 0) > 0 or T < 16:
            from .reference import ReferenceBackend
            from ...dsl.ast import GatherContext, Program
            ref = ReferenceBackend(device=self.device, dtype=q.dtype)
            prog = Program()
            prog.add_stmt("y", GatherContext(
                query=op["query"], source=op["source"],
                top_k=op.get("top_k", 0), causal=op.get("causal", True)
            ))
            return ref.execute(prog, ws)["y"]

        output = torch.zeros(B, T, D, device=self.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(T, meta["BLOCK_T"]),)
        _gather_context_kernel[grid](
            q,
            src,
            output,
            T=T,
            D=D,
            BLOCK_T=min(32, T),
            BLOCK_D=min(128, D),
        )

        if sq:
            output = output.squeeze(0)
        return output.to(dtype=ws[op["query"]].dtype)

    def _run_query_memory(self, op, ws, batch_size) -> torch.Tensor:
        x = ws[op["input"]].contiguous()
        db_key = op["db"]
        top_k = op["top_k"]

        if db_key not in self.stdlib_weights:
            return x

        db = self.stdlib_weights[db_key]
        keys = db.get("keys")
        values = db.get("values")
        if keys is None or values is None:
            return x

        keys = keys.to(device=x.device, dtype=torch.float32)
        values = values.to(device=x.device, dtype=torch.float32)
        x_f = x.to(dtype=torch.float32)
        if x_f.dim() != 2:
            x_f = x_f.reshape(-1, x_f.shape[-1])

        scores = x_f @ keys.T
        k = min(top_k, keys.shape[0])
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)
        weights = torch.nn.functional.softmax(topk_vals, dim=-1)
        result_flat = (weights.unsqueeze(-1) * values[topk_idx]).sum(dim=-2)

        if x.dim() == 3:
            result_flat = result_flat.reshape(x.shape[0], x.shape[1], -1)
        elif x.dim() == 2:
            result_flat = result_flat.reshape(x.shape[0], -1)

        return result_flat.to(dtype=x.dtype)

    def _apply_transform_gpu(self, op, ws) -> torch.Tensor:
        x = ws[op["input"]].contiguous()
        d_in = x.shape[-1]

        matrix_ref = op["matrix"]
        if matrix_ref.ref_type == "stdlib" and matrix_ref.name in self.stdlib_weights:
            entry = self.stdlib_weights[matrix_ref.name]
            opt = entry.get("operator_type", "dense")

            if opt == "multihead_attention":
                from .reference import ReferenceBackend
                from ...dsl.ast import Program, Transform, MatrixRef
                ref = ReferenceBackend(stdlib_weights=self.stdlib_weights, device=self.device, dtype=x.dtype)
                prog = Program()
                prog.add_stmt("y", Transform(matrix_ref.name, MatrixRef("stdlib", matrix_ref.name)))
                return ref.execute(prog, {"x": x})["y"]

            if opt == "sparse_down_projection" or opt == "sparse_down_projection_lr" or opt == "gated_sparse_mlp":
                from .reference import ReferenceBackend
                ref = ReferenceBackend(stdlib_weights=self.stdlib_weights, device=self.device, dtype=x.dtype)
                prog = Program()
                from ...dsl.ast import Transform, MatrixRef
                prog.add_stmt("y", Transform(matrix_ref.name, MatrixRef("stdlib", matrix_ref.name)))
                return ref.execute(prog, {"x": x})["y"]

            if opt == "direction_vector":
                vec = entry["vector"].to(device=x.device, dtype=x.dtype)
                return x + vec

            w = self._get_transform_weight(matrix_ref.name, d_in)
            w = w.to(device=x.device, dtype=torch.float32).contiguous()
            return torch.nn.functional.linear(x.float(), w).to(dtype=x.dtype)
        else:
            return x

    def _get_transform_weight(
        self, name: str, d_model: int
    ) -> torch.Tensor:
        entry = self.stdlib_weights[name]
        opt = entry.get("operator_type", "dense")

        if opt == "low_rank_projection":
            u = entry["u"].to(dtype=torch.float32)
            v = entry["v"].to(dtype=torch.float32)
            return (u.T @ v)

        elif opt == "direction_vector":
            return entry["vector"].to(dtype=torch.float32)

        else:
            w = entry.get("weight")
            if w is not None:
                return w.to(dtype=torch.float32)
            return torch.eye(d_model, dtype=torch.float32)
