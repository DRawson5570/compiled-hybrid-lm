from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..backend.jit_compiler import JITCompiler
from ..frontend.meta_compiler import MetaCompiler


def reinforce_update(
    meta_compiler: MetaCompiler,
    embeddings: torch.Tensor,
    reward: float,
    baseline: float,
    optimizer: torch.optim.Optimizer,
    stdlib_names: List[str] | None = None,
    maximize: bool = True,
) -> Dict[str, float]:
    meta_compiler.train()
    optimizer.zero_grad(set_to_none=True)

    z = meta_compiler.context_analyzer(embeddings)

    template_logits = meta_compiler.template_selector(z)
    template_probs = F.softmax(template_logits, dim=-1)
    template_id = int(template_probs.argmax(dim=-1).item())

    log_prob = torch.log(template_probs[0, template_id] + 1e-8)
    advantage = reward - baseline if maximize else baseline - reward

    reinforce_loss = -log_prob * advantage

    pred_params = meta_compiler.parameter_generator(z)
    if pred_params.shape[0] > 1:
        pred_params = pred_params.mean(dim=0, keepdim=True)

    loss = reinforce_loss + 0.001 * pred_params.sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(meta_compiler.trainable_parameters(), 1.0)
    optimizer.step()

    return {
        "template_id": template_id,
        "log_prob": float(log_prob.item()),
        "advantage": float(advantage),
        "reward": float(reward),
        "loss": float(loss.item()),
    }


def train_with_reinforce(
    meta_compiler: MetaCompiler,
    reward_fn: Callable,
    data_loader: List[torch.Tensor],
    steps: int = 1000,
    lr: float = 1e-3,
    baseline_alpha: float = 0.9,
    stdlib_names: List[str] | None = None,
    maximize: bool = True,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        meta_compiler.trainable_parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    baseline = 0.0
    history = []

    for step in range(steps):
        idx = torch.randint(0, len(data_loader), (1,)).item()
        embeddings = data_loader[idx]

        meta_compiler.eval()
        with torch.no_grad():
            program = meta_compiler.synthesize(
                embeddings, stdlib_names=stdlib_names
            )
        loss_value = reward_fn(embeddings, program)

        meta_compiler.train()
        metrics = reinforce_update(
            meta_compiler,
            embeddings,
            float(loss_value),
            baseline,
            optimizer,
            stdlib_names=stdlib_names,
            maximize=maximize,
        )

        baseline = baseline_alpha * baseline + (1 - baseline_alpha) * loss_value
        metrics["reward"] = float(loss_value)
        metrics["baseline"] = float(baseline)
        history.append(metrics)

        if verbose and (step == 0 or (step + 1) % 100 == 0 or step == steps - 1):
            print(
                f"  REINFORCE step {step+1:5d}/{steps}  "
                f"reward={loss_value:.4f}  baseline={baseline:.4f}  "
                f"adv={metrics['advantage']:.4f}",
                flush=True,
            )

    return history
