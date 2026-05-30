from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ..backend.jit_compiler import JITCompiler, KernelHandle
from ..dsl.ast import Program
from .workspace import TensorWorkspace


class UCNExecutor:
    def __init__(
        self,
        d_model: int,
        stdlib_weights: Dict[str, Any] | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        use_triton: bool = False,
        max_tokens: int = 512,
    ):
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.compiler = JITCompiler(
            stdlib_weights=stdlib_weights,
            device=device,
            dtype=dtype,
            use_triton=use_triton,
        )

        self.workspace = TensorWorkspace(
            max_tokens=max_tokens,
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        self._program_cache: Dict[int, KernelHandle] = {}

    def forward(
        self,
        token_embeddings: torch.Tensor,
        program: Optional[Program] = None,
        context_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if token_embeddings.dim() == 2:
            token_embeddings = token_embeddings.unsqueeze(0)

        batch_size, seq_len, d_in = token_embeddings.shape
        if d_in != self.d_model:
            raise ValueError(
                f"Input dim {d_in} != d_model {self.d_model}"
            )

        if program is None:
            program = self._build_identity_program()

        self.workspace.clear()
        self.workspace.set("input", token_embeddings)

        inputs = {"input": token_embeddings}
        outputs = self.compiler.compile_and_execute(
            program,
            inputs,
            batch_size=batch_size,
            context_z=context_z,
        )

        result_name = program.statements[-1].target if program.statements else "input"
        return outputs.get(result_name, token_embeddings)

    def execute_raw(
        self,
        program: Program,
        inputs: Dict[str, torch.Tensor],
        batch_size: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self.compiler.compile_and_execute(program, inputs, batch_size)

    def _build_identity_program(self) -> Program:
        from ..dsl.ast import Activate, ActivateType, Program, Statement

        program = Program()
        program.add_stmt("output", Activate("input", ActivateType.IDENTITY))
        return program

    def clear_caches(self):
        self.compiler.clear_caches()
        self.workspace.clear()
        self._program_cache.clear()


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = x.dtype
    x_f = x.to(dtype=torch.float32)
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * rms * weight.to(dtype=torch.float32)).to(dtype=dtype)


class MultiLayerUCNExecutor:
    def __init__(
        self,
        d_model: int,
        stdlib_weights: Dict[str, Any] | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        use_triton: bool = False,
        max_tokens: int = 512,
    ):
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.stdlib_weights = stdlib_weights or {}
        self.use_triton = use_triton

        self.executor = UCNExecutor(
            d_model=d_model,
            stdlib_weights=stdlib_weights,
            device=device,
            dtype=dtype,
            use_triton=use_triton,
            max_tokens=max_tokens,
        )

        self._norm_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self._programs: Dict[int, Dict[str, Program]] = {}
        self._corrections: Dict[int, torch.Tensor] = {}

    def set_norm_weights(
        self,
        layer_idx: int,
        input_layernorm_weight: torch.Tensor,
        post_attention_layernorm_weight: torch.Tensor,
    ):
        self._norm_weights[layer_idx] = {
            "pre_attn": input_layernorm_weight.to(device=self.device, dtype=self.dtype),
            "pre_mlp": post_attention_layernorm_weight.to(device=self.device, dtype=self.dtype),
        }

    def set_layer_programs(
        self,
        layer_idx: int,
        program_attn: Program,
        program_mlp: Program,
    ):
        self._programs[layer_idx] = {
            "attn": program_attn,
            "mlp": program_mlp,
        }

    def compute_correction_bias(
        self,
        model,
        tokenizer,
        prompts: List[str],
        layers: List[int],
        device: str = "cuda",
    ):
        """
        Compute per-layer residual correction: mean(real_mlp - ucn_mlp)
        on calibration prompts. This compensates for systematic sparse MLP error.
        """
        corrections: Dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            errors = []

            mlp = model.model.layers[layer_idx].mlp
            layer = model.model.layers[layer_idx]

            mlp_inputs = []
            mlp_outputs = []

            def pre_h(module, args, kwargs):
                if args: mlp_inputs.append(args[0].detach().cpu())
                elif "hidden_states" in kwargs: mlp_inputs.append(kwargs["hidden_states"].detach().cpu())

            def post_h(module, args, output):
                mlp_outputs.append((output[0] if isinstance(output, tuple) else output).detach().cpu())

            ph = mlp.register_forward_pre_hook(pre_h, with_kwargs=True)
            pth = mlp.register_forward_hook(post_h)

            for prompt in prompts[:8]:
                inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
                with torch.no_grad():
                    model(**inp)

            pth.remove(); ph.remove()

            if not mlp_inputs or not mlp_outputs:
                corrections[layer_idx] = torch.zeros(self.d_model)
                continue

            x = torch.cat([t.reshape(-1, t.shape[-1]) for t in mlp_inputs], dim=0)[:64].float()
            y_real = torch.cat([t.reshape(-1, t.shape[-1]) for t in mlp_outputs], dim=0)[:64].float()

            prog = self._programs.get(layer_idx, {}).get("mlp")
            if prog is None:
                corrections[layer_idx] = torch.zeros(self.d_model)
                continue

            y_ucn = self.executor.execute_raw(prog, {"x": x.cpu()}, batch_size=min(64, x.shape[0])).get("y", torch.zeros_like(x))

            error = (y_real - y_ucn).mean(dim=0)
            corrections[layer_idx] = error

        self._corrections = corrections
        return corrections

    def forward_layers(
        self,
        embeddings: torch.Tensor,
        layers: List[int] | None = None,
        mode: str = "delta",
    ) -> torch.Tensor:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)

        h = embeddings.to(device=self.device, dtype=self.dtype)
        if layers is None:
            layers = sorted(self._programs.keys())

        for layer_idx in layers:
            if layer_idx not in self._programs:
                continue

            norms = self._norm_weights.get(layer_idx)
            progs = self._programs[layer_idx]
            correction = self._corrections.get(layer_idx)

            if norms:
                h_norm = rms_norm(h, norms["pre_attn"])
            else:
                h_norm = h

            attn_out = self.executor.execute_raw(
                progs["attn"], {"x": h_norm}
            ).get("y", torch.zeros_like(h))
            h = h + attn_out

            if norms:
                h_norm = rms_norm(h, norms["pre_mlp"])
            else:
                h_norm = h

            ucn_mlp = self.executor.execute_raw(
                progs["mlp"], {"x": h_norm}
            ).get("y", torch.zeros_like(h))

            if mode == "delta":
                delta = ucn_mlp - h_norm
                ucn_mlp = h_norm + delta

            if correction is not None and correction.abs().sum() > 0:
                ucn_mlp = ucn_mlp + correction.to(device=self.device, dtype=self.dtype)

            if mode == "delta":
                h = h_norm + (ucn_mlp - h_norm)
            else:
                h = h + ucn_mlp

        return h
