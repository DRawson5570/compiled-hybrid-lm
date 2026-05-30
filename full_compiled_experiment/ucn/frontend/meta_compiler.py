from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..dsl.ast import Program
from .context_analyzer import ContextAnalyzer
from .parameter_generator import ParameterGenerator
from .template_library import TemplateLibrary
from .template_selector import TemplateSelector


class MetaCompiler:
    def __init__(
        self,
        d_model: int,
        n_templates: int | None = None,
        max_params: int = 8,
        d_latent: int = 128,
        n_layers: int = 2,
        device: str = "cuda",
    ):
        self.d_model = d_model
        self.d_latent = d_latent

        self.template_library = TemplateLibrary()
        n_templates = n_templates or self.template_library.n_templates

        self.context_analyzer = ContextAnalyzer(
            d_model=d_model,
            d_latent=d_latent,
            n_layers=n_layers,
        )

        self.template_selector = TemplateSelector(
            d_latent=d_latent,
            n_templates=n_templates,
        )

        self.parameter_generator = ParameterGenerator(
            d_latent=d_latent,
            n_params=max_params,
        )

        self.device = device
        self.to(device)

    def to(self, device: str | torch.device):
        self.context_analyzer.to(device)
        self.template_selector.to(device)
        self.parameter_generator.to(device)
        if isinstance(device, str):
            self.device = device
        else:
            self.device = str(device)
        return self

    def forward(
        self,
        embeddings: torch.Tensor,
        stdlib_names: List[str] | None = None,
        temperature: float = 1.0,
    ) -> Program:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.to(self.device)

        z = self.context_analyzer(embeddings)

        template_logits = self.template_selector(z, temperature=temperature)
        if template_logits.dim() > 1 and template_logits.shape[0] > 1:
            template_logits = template_logits.mean(dim=0, keepdim=True)
        template_id = int(template_logits.argmax(dim=-1).item())

        params = self.parameter_generator(z)
        if params.dim() > 1 and params.shape[0] > 1:
            params = params.mean(dim=0, keepdim=True)
        param_list = params[0].tolist()

        program = self.template_library.build_program(
            template_id=template_id,
            params=param_list,
            stdlib_names=stdlib_names,
        )

        return program

    def synthesize(
        self,
        embeddings: torch.Tensor,
        stdlib_names: List[str] | None = None,
        temperature: float = 1.0,
    ) -> Program:
        return self.forward(embeddings, stdlib_names, temperature)

    def sample_program(
        self,
        embeddings: torch.Tensor,
        stdlib_names: List[str] | None = None,
        temperature: float = 1.0,
    ) -> Program:
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.to(self.device)

        z = self.context_analyzer(embeddings)

        template_probs = self.template_selector.sample_gumbel(
            z, temperature=temperature, hard=True
        )
        template_id = int(template_probs.argmax(dim=-1).item())

        params = self.parameter_generator(z)
        param_list = params[0].tolist()

        program = self.template_library.build_program(
            template_id=template_id,
            params=param_list,
            stdlib_names=stdlib_names,
        )

        return program

    def synthesize_soft_forward(
        self,
        x: torch.Tensor,
        stdlib_names: List[str],
        executor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Differentiable soft synthesis. Attaches MetaCompiler predictions to output
        via a soft combination anchored on x, ensuring gradient flow.
        """
        if hasattr(executor, 'stdlib_weights'):
            stdlib = executor.stdlib_weights
        elif hasattr(executor, 'compiler') and hasattr(executor.compiler, 'stdlib_weights'):
            stdlib = executor.compiler.stdlib_weights
        else:
            stdlib = {}

        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeezed = True
        else:
            squeezed = False

        x = x.to(dtype=torch.float32)
        z = self.context_analyzer(x)
        template_weights = F.softmax(self.template_selector(z, temperature=temperature), dim=-1)[0]

        n_tmpl = template_weights.shape[0]
        n_lib = len(stdlib_names)

        from ..dsl.ast import MatrixRef, Program, Transform

        x_flat = x.reshape(-1, x.shape[-1])
        base_input = x_flat.detach()
        result = torch.zeros_like(x_flat, dtype=torch.float32)

        for tid in range(n_tmpl):
            w = template_weights[tid]
            lib_idx = tid % n_lib
            if lib_idx < n_lib:
                entry_name = stdlib_names[lib_idx]
                if entry_name in stdlib:
                    prog = Program()
                    prog.add_stmt("y", Transform("x", MatrixRef("stdlib", entry_name)))
                    out = executor.execute(prog, {"x": base_input}, batch_size=x_flat.shape[0])["y"]
                    result = result + w.unsqueeze(-1) * out.to(dtype=torch.float32, device=result.device)

        result = base_input + result

        if squeezed:
            result = result.squeeze(0)

        return result.to(dtype=torch.float32)

    def train(self, mode: bool = True):
        self.context_analyzer.train(mode)
        self.template_selector.train(mode)
        self.parameter_generator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        for module in [
            self.context_analyzer,
            self.template_selector,
            self.parameter_generator,
        ]:
            yield from module.parameters()

    def trainable_parameters(self):
        modules = [self.context_analyzer, self.template_selector, self.parameter_generator]
        return [p for m in modules for p in m.parameters() if p.requires_grad]

    def state_dict(self) -> Dict:
        return {
            "context_analyzer": self.context_analyzer.state_dict(),
            "template_selector": self.template_selector.state_dict(),
            "parameter_generator": self.parameter_generator.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict, strict: bool = True):
        self.context_analyzer.load_state_dict(
            state_dict["context_analyzer"], strict=strict
        )
        self.template_selector.load_state_dict(
            state_dict["template_selector"], strict=strict
        )
        self.parameter_generator.load_state_dict(
            state_dict["parameter_generator"], strict=strict
        )
