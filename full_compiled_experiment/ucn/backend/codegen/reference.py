from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ...dsl.ast import (
    Activate,
    ActivateType,
    AllocDecl,
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
from ...dsl.types import Matrix


class ReferenceBackend:
    def __init__(
        self,
        stdlib_weights: Dict[str, Any] | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.stdlib_weights = stdlib_weights or {}
        self.device = device
        self.dtype = dtype

    def execute(
        self,
        program: Program,
        inputs: Dict[str, torch.Tensor],
        batch_size: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        workspace: Dict[str, torch.Tensor] = dict(inputs)

        for decl in program.declarations:
            if decl.name not in workspace:
                workspace[decl.name] = self._allocate(decl, batch_size)

        for stmt in program.statements:
            result = self._execute_stmt(stmt, workspace)
            workspace[stmt.target] = result

        return workspace

    def _allocate(self, decl: AllocDecl, batch_size: int | None = None) -> torch.Tensor:
        if decl.type_spec.value == "Vector":
            shape = (batch_size, decl.dim) if batch_size else (decl.dim,)
            return torch.zeros(shape, device=self.device, dtype=self.dtype)
        return torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def _execute_stmt(self, stmt: Statement, ws: Dict[str, torch.Tensor]) -> torch.Tensor:
        expr = stmt.expr
        if isinstance(expr, Mix):
            return self._execute_mix(expr, ws)
        elif isinstance(expr, Project):
            return self._execute_project(expr, ws)
        elif isinstance(expr, Transform):
            return self._execute_transform(expr, ws)
        elif isinstance(expr, Activate):
            return self._execute_activate(expr, ws)
        elif isinstance(expr, QueryMemory):
            return self._execute_query_memory(expr, ws)
        elif isinstance(expr, Residual):
            return self._execute_residual(expr, ws)
        elif isinstance(expr, Rotate):
            return self._execute_rotate(expr, ws)
        elif isinstance(expr, GatherContext):
            return self._execute_gather_context(expr, ws)
        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")

    def _execute_mix(self, expr: Mix, ws: Dict[str, torch.Tensor]) -> torch.Tensor:
        result = None
        for name, weight in zip(expr.inputs, expr.weights):
            x = ws[name]
            if result is None:
                result = weight * x
            else:
                result = result + weight * x
        return result

    def _execute_project(self, expr: Project, ws: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = ws[expr.input]
        ss = expr.subspace
        result = torch.zeros_like(x)
        if x.dim() == 2:
            result[:, ss.start : ss.end] = x[:, ss.start : ss.end]
        else:
            result[ss.start : ss.end] = x[ss.start : ss.end]
        return result

    def _execute_transform(self, expr: Transform, ws: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = ws[expr.input]

        if expr.matrix.ref_type == "stdlib" and expr.matrix.name in self.stdlib_weights:
            weights = self.stdlib_weights[expr.matrix.name]
            return self._apply_transform(x, weights)
        elif expr.matrix.ref_type == "dynamic":
            return self._apply_dynamic_transform(x, expr.matrix.name, ws)
        else:
            raise KeyError(
                f"Matrix {expr.matrix.ref_type}.{expr.matrix.name} not found in stdlib"
            )

    def _apply_transform(
        self, x: torch.Tensor, weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        opt = weights.get("operator_type", "dense")
        if opt == "low_rank_projection":
            u = weights["u"].to(device=x.device, dtype=x.dtype)
            v = weights["v"].to(device=x.device, dtype=x.dtype)
            if x.dim() == 2:
                mid = x @ u.T
                return mid @ v
            else:
                mid = x @ u.T
                return mid @ v
        elif opt == "multihead_attention":
            return self._apply_full_attention(x, weights)
        elif opt == "gated_sparse_mlp":
            return self._apply_gated_sparse_mlp(x, weights)
        elif opt == "sparse_down_projection":
            return self._apply_sparse_down_projection(x, weights)
        elif opt == "sparse_down_projection_lr":
            return self._apply_sparse_down_lr(x, weights)
        elif opt == "direction_vector":
            vec = weights["vector"].to(device=x.device, dtype=x.dtype)
            return x + vec
        else:
            w = weights.get("weight")
            if w is not None:
                w = w.to(device=x.device, dtype=x.dtype)
                return x @ w.T
            return x

    def _apply_full_attention(
        self, x: torch.Tensor, weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        W_q = weights["W_q"].to(device=x.device, dtype=torch.float32)
        W_k = weights["W_k"].to(device=x.device, dtype=torch.float32)
        W_v = weights["W_v"].to(device=x.device, dtype=torch.float32)
        W_o = weights["W_o"].to(device=x.device, dtype=torch.float32)

        b_q = weights.get("b_q")
        b_k = weights.get("b_k")
        b_v = weights.get("b_v")

        if b_q is not None:
            b_q = b_q.to(device=x.device, dtype=torch.float32)
        if b_k is not None:
            b_k = b_k.to(device=x.device, dtype=torch.float32)
        if b_v is not None:
            b_v = b_v.to(device=x.device, dtype=torch.float32)

        n_heads = int(weights.get("n_heads", 12))
        n_kv_heads = int(weights.get("n_kv_heads", 2))
        head_dim = int(weights.get("head_dim", 128))
        n_groups = n_heads // n_kv_heads

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        B, T, D = x.shape
        x = x.to(dtype=torch.float32)

        q = x @ W_q.T
        if b_q is not None:
            q = q + b_q
        k = x @ W_k.T
        if b_k is not None:
            k = k + b_k
        v = x @ W_v.T
        if b_v is not None:
            v = v + b_v

        q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, n_kv_heads, head_dim).transpose(1, 2)

        cos = weights.get("cos")
        sin = weights.get("sin")
        if cos is not None and sin is not None:
            cos = cos.to(device=x.device, dtype=torch.float32)
            sin = sin.to(device=x.device, dtype=torch.float32)
            if cos.dim() == 2:
                cos = cos.unsqueeze(0)
            if sin.dim() == 2:
                sin = sin.unsqueeze(0)
            if cos.shape[1] < T:
                cos = torch.nn.functional.interpolate(
                    cos.transpose(1, 2), size=T, mode="linear"
                ).transpose(1, 2)
                sin = torch.nn.functional.interpolate(
                    sin.transpose(1, 2), size=T, mode="linear"
                ).transpose(1, 2)
            cos = cos[:, :T, :]
            sin = sin[:, :T, :]

            half = head_dim // 2
            q_rot = torch.zeros_like(q)
            k_rot = torch.zeros_like(k)
            q_rot[..., :half] = q[..., :half] * cos[..., :half] - q[..., half:] * sin[..., :half]
            q_rot[..., half:] = q[..., :half] * sin[..., :half] + q[..., half:] * cos[..., :half]
            k_rot[..., :half] = k[..., :half] * cos[..., :half] - k[..., half:] * sin[..., :half]
            k_rot[..., half:] = k[..., :half] * sin[..., :half] + k[..., half:] * cos[..., :half]
            q = q_rot
            k = k_rot

        k = k.unsqueeze(2).expand(B, n_kv_heads, n_groups, T, head_dim).reshape(B, n_heads, T, head_dim)
        v = v.unsqueeze(2).expand(B, n_kv_heads, n_groups, T, head_dim).reshape(B, n_heads, T, head_dim)

        import torch.nn.functional as F

        scale = head_dim ** -0.5
        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=scale
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, D)

        result = attn_output @ W_o.T

        if squeezed:
            result = result.squeeze(0)

        return result.to(dtype=x.dtype)

    def _apply_gated_sparse_mlp(
        self, x: torch.Tensor, weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        gate_keys = weights["gate_keys"].to(device=x.device, dtype=torch.float32)
        gate_biases = weights["gate_biases"].to(device=x.device, dtype=torch.float32)
        values = weights["values"].to(device=x.device, dtype=torch.float32)
        scales = weights.get("scales")

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        B, T, D = x.shape
        x_flat = x.reshape(-1, D).to(dtype=torch.float32)

        gate_pre = x_flat @ gate_keys.T + gate_biases
        gate_post = F.silu(gate_pre)

        if scales is not None:
            scales = scales.to(device=x.device, dtype=torch.float32)
            gate_post = gate_post * scales

        result_flat = gate_post @ values
        if squeezed:
            result_flat = result_flat.squeeze(0)
        else:
            result_flat = result_flat.reshape(B, T, -1)

        return result_flat.to(dtype=x.dtype)

    def _apply_sparse_down_projection(
        self, x: torch.Tensor, weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        gate_w = weights["gate_weight"].to(device=x.device, dtype=torch.float32)
        up_w = weights["up_weight"].to(device=x.device, dtype=torch.float32)
        down_w = weights["down_weight"].to(device=x.device, dtype=torch.float32)
        gate_b = weights.get("gate_bias")
        if gate_b is not None:
            gate_b = gate_b.to(device=x.device, dtype=torch.float32)
        top_k = int(weights.get("top_k", 256))

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        B, T, D = x.shape
        x_flat = x.reshape(-1, D).to(dtype=torch.float32)
        N = x_flat.shape[0]
        n_neurons = gate_w.shape[0]

        gate_pre = x_flat @ gate_w.T
        if gate_b is not None:
            gate_pre = gate_pre + gate_b
        gate_post = F.silu(gate_pre)

        up_out = x_flat @ up_w.T
        contributions = gate_post * up_out

        k = min(top_k, n_neurons)
        col_norms = down_w.norm(dim=0).to(device=x.device)
        weighted = contributions * col_norms
        _, topk_idx = torch.topk(weighted, k=k, dim=-1)

        down_selected = down_w[:, topk_idx.long()].permute(1, 2, 0)
        topk_gate = contributions.gather(-1, topk_idx)

        result_flat = torch.bmm(
            topk_gate.unsqueeze(1), down_selected
        ).squeeze(1)

        if squeezed:
            result_flat = result_flat.squeeze(0)
        else:
            result_flat = result_flat.reshape(B, T, -1)

        return result_flat.to(dtype=x.dtype)

    def _apply_sparse_down_lr(
        self, x: torch.Tensor, weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        gate_w = weights["gate_weight"].to(device=x.device, dtype=torch.float32)
        up_w = weights["up_weight"].to(device=x.device, dtype=torch.float32)
        down_w = weights["down_weight"].to(device=x.device, dtype=torch.float32)
        gate_b = weights.get("gate_bias")
        if gate_b is not None:
            gate_b = gate_b.to(device=x.device, dtype=torch.float32)
        top_k = int(weights.get("top_k", 256))

        V_r = weights.get("V_r")
        U_r = weights.get("U_r")
        if V_r is not None:
            V_r = V_r.to(device=x.device, dtype=torch.float32)
        if U_r is not None:
            U_r = U_r.to(device=x.device, dtype=torch.float32)

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        B, T, D = x.shape
        x_flat = x.reshape(-1, D).to(dtype=torch.float32)
        N = x_flat.shape[0]
        n_neurons = gate_w.shape[0]

        gate_pre = x_flat @ gate_w.T
        if gate_b is not None:
            gate_pre = gate_pre + gate_b
        gate_post = F.silu(gate_pre)
        up_out = x_flat @ up_w.T
        contributions = gate_post * up_out

        k = min(top_k, n_neurons)
        col_norms = down_w.norm(dim=0).to(device=x.device)
        weighted = contributions * col_norms
        _, topk_idx = torch.topk(weighted, k=k, dim=-1)

        down_selected = down_w[:, topk_idx.long()].permute(1, 2, 0)
        topk_gate = contributions.gather(-1, topk_idx)

        result_flat = torch.bmm(
            topk_gate.unsqueeze(1), down_selected
        ).squeeze(1)

        if V_r is not None and U_r is not None:
            contrib_mask = torch.ones_like(contributions)
            contrib_mask.scatter_(-1, topk_idx, 0)
            contrib_rest = contributions * contrib_mask
            correction = (contrib_rest @ V_r.T) @ U_r.T
            result_flat = result_flat + correction

        if squeezed:
            result_flat = result_flat.squeeze(0)
        else:
            result_flat = result_flat.reshape(B, T, -1)

        return result_flat.to(dtype=x.dtype)

    def _apply_dynamic_transform(
        self, x: torch.Tensor, name: str, ws: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if name in ws:
            return ws[name]
        return x

    def _execute_activate(
        self, expr: Activate, ws: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = ws[expr.input]
        if expr.activation == ActivateType.GELU:
            return F.gelu(x)
        elif expr.activation == ActivateType.RELU:
            return F.relu(x)
        elif expr.activation == ActivateType.SILU:
            return F.silu(x)
        else:
            return x

    def _execute_query_memory(
        self, expr: QueryMemory, ws: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = ws[expr.input]
        db_key = expr.db.partition

        if db_key in self.stdlib_weights:
            db = self.stdlib_weights[db_key]
            return self._sparse_lookup(x, db, expr.top_k)
        return x

    def _sparse_lookup(
        self, x: torch.Tensor, db: Any, top_k: int
    ) -> torch.Tensor:
        keys = db.get("keys")
        values = db.get("values")
        if keys is None or values is None:
            return x

        keys = keys.to(device=x.device, dtype=x.dtype)
        values = values.to(device=x.device, dtype=x.dtype)

        if x.dim() == 2:
            scores = x @ keys.T
            topk_vals, topk_idx = torch.topk(scores, k=min(top_k, keys.shape[0]), dim=-1)
            weights = F.softmax(topk_vals, dim=-1)
            return (weights.unsqueeze(-1) * values[topk_idx]).sum(dim=-2)
        else:
            scores = x @ keys.T
            topk_vals, topk_idx = torch.topk(scores, k=min(top_k, keys.shape[0]))
            weights = F.softmax(topk_vals, dim=-1)
            return (weights.unsqueeze(-1) * values[topk_idx]).sum(dim=-2)

    def _execute_residual(
        self, expr: Residual, ws: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if not expr.inputs:
            raise ValueError("residual requires at least one input")
        result = ws[expr.inputs[0]].clone()
        for name in expr.inputs[1:]:
            result = result + ws[name]
        return result

    def _execute_rotate(
        self, expr: Rotate, ws: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        x = ws[expr.input]
        ss = expr.subspace
        theta = expr.theta

        cos_theta = torch.cos(torch.tensor(theta, device=x.device, dtype=torch.float32))
        sin_theta = torch.sin(torch.tensor(theta, device=x.device, dtype=torch.float32))

        result = x.clone().to(dtype=torch.float32)
        half = ss.size // 2

        if half <= 0:
            return x

        max_idx = min(ss.start + half * 2, x.shape[-1])
        if max_idx <= ss.start:
            return x

        if x.dim() == 2:
            d0 = result[:, ss.start : ss.start + half].clone()
            d1 = result[:, ss.start + half : max_idx].clone()
            d0_half_len = d0.shape[1]
            d1_half_len = d1.shape[1]
            common_len = min(d0_half_len, d1_half_len)
            result[:, ss.start : ss.start + common_len] = (
                d0[:, :common_len] * cos_theta - d1[:, :common_len] * sin_theta
            )
            result[:, ss.start + half : ss.start + half + common_len] = (
                d0[:, :common_len] * sin_theta + d1[:, :common_len] * cos_theta
            )
        else:
            d0 = result[ss.start : ss.start + half].clone()
            d1 = result[ss.start + half : max_idx].clone()
            d0_len = d0.shape[0]
            d1_len = d1.shape[0]
            common_len = min(d0_len, d1_len)
            result[ss.start : ss.start + common_len] = (
                d0[:common_len] * cos_theta - d1[:common_len] * sin_theta
            )
            result[ss.start + half : ss.start + half + common_len] = (
                d0[:common_len] * sin_theta + d1[:common_len] * cos_theta
            )

        return result.to(dtype=x.dtype)

    def _execute_gather_context(
        self, expr: GatherContext, ws: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        q = ws[expr.query].to(dtype=torch.float32)
        src = ws[expr.source].to(dtype=torch.float32)

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

        B, Tq, D = q.shape
        _, Ts, _ = src.shape
        scale = D ** -0.5

        scores = (q @ src.transpose(-2, -1)) * scale

        if expr.causal:
            mask = torch.triu(
                torch.full((Tq, Ts), float("-inf"), device=q.device, dtype=torch.float32),
                diagonal=1,
            )
            scores = scores + mask

        if expr.top_k > 0 and expr.top_k < Ts:
            topk_vals, topk_idx = torch.topk(scores, k=expr.top_k, dim=-1)
            mask_val = torch.full_like(scores, float("-inf"))
            mask_val.scatter_(-1, topk_idx, topk_vals)
            scores = mask_val

        weights = F.softmax(scores, dim=-1)
        result = weights @ src

        if sq:
            result = result.squeeze(0)
        if ss and result.dim() == 3:
            result = result.squeeze(0)

        return result.to(dtype=ws[expr.query].dtype)
