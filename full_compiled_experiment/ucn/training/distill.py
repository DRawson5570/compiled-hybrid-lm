from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..backend.jit_compiler import JITCompiler
from ..frontend.meta_compiler import MetaCompiler


def distill_step(
    meta_compiler: MetaCompiler,
    teacher_embeddings: torch.Tensor,
    target_template: int,
    target_params: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    meta_compiler.train()
    optimizer.zero_grad(set_to_none=True)

    z = meta_compiler.context_analyzer(teacher_embeddings)

    template_logits = meta_compiler.template_selector(z)
    template_loss = F.cross_entropy(
        template_logits,
        torch.tensor([target_template], device=template_logits.device),
    )

    pred_params = meta_compiler.parameter_generator(z)
    if pred_params.shape[0] > 1:
        pred_params = pred_params.mean(dim=0, keepdim=True)
    param_loss = F.mse_loss(pred_params[0], target_params)

    loss = template_loss + 0.1 * param_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(meta_compiler.trainable_parameters(), 1.0)
    optimizer.step()

    return {
        "template_loss": float(template_loss.item()),
        "param_loss": float(param_loss.item()),
        "total_loss": float(loss.item()),
    }


def train_meta_compiler_supervised(
    meta_compiler: MetaCompiler,
    distillation_data: List[Tuple[torch.Tensor, int, torch.Tensor]],
    steps: int = 1000,
    lr: float = 1e-3,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    if not distillation_data:
        return []

    optimizer = torch.optim.AdamW(
        meta_compiler.trainable_parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    history = []

    for step in range(steps):
        idx = torch.randint(0, len(distillation_data), (1,)).item()
        emb, template_id, params = distillation_data[idx]

        metrics = distill_step(
            meta_compiler, emb, template_id, params, optimizer
        )
        history.append(metrics)

        if verbose and (step == 0 or (step + 1) % 100 == 0 or step == steps - 1):
            print(
                f"  Distill step {step+1:5d}/{steps}  "
                f"t_loss={metrics['template_loss']:.4f}  "
                f"p_loss={metrics['param_loss']:.4f}  "
                f"total={metrics['total_loss']:.4f}",
                flush=True,
            )

    return history


def evaluate_meta_compiler(
    meta_compiler: MetaCompiler,
    eval_data: List[Tuple[torch.Tensor, int, torch.Tensor]],
) -> Dict[str, float]:
    meta_compiler.eval()
    correct = 0
    total_param_loss = 0.0

    with torch.no_grad():
        for emb, target_template, target_params in eval_data:
            z = meta_compiler.context_analyzer(emb)
            pred_template = int(
                meta_compiler.template_selector(z).argmax(dim=-1).item()
            )
            if pred_template == target_template:
                correct += 1

            pred_params = meta_compiler.parameter_generator(z)[0]
            total_param_loss += float(
                F.mse_loss(pred_params, target_params).item()
            )

    return {
        "template_accuracy": correct / len(eval_data),
        "avg_param_mse": total_param_loss / len(eval_data),
    }
