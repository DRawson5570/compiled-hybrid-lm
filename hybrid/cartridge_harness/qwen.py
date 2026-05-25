"""Qwen cartridge runner and trainer for owned cartridge research loops."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from hybrid.cartridge_harness.core import (
    TaskExample,
    build_summary,
    compare_rows,
    evaluate_text_runner,
)
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer


class QwenAdapterCartridgeRunner:
    """Frozen Qwen plus a trainable feature-conditioned cartridge."""

    def __init__(self, model_name: str, device: str = "cuda", bottleneck: int = 64,
                 cartridge_id: str = "owned-qwen-adapter-cartridge",
                 role: str | CartridgeRole = CartridgeRole.DOMAIN_CAPABILITY,
                 source_corpus: str = "hybrid.cartridge_harness"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self.hf_model.eval()
        for param in self.hf_model.parameters():
            param.requires_grad = False

        self.d_model = self.hf_model.config.hidden_size
        self.inject_layers = [
            idx for idx in (0, 2, 4, 7, 10, 14, 17, 21, 24, 26)
            if idx < len(self.hf_model.model.layers)
        ]
        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=self.d_model,
            inject_layers=self.inject_layers,
            bottleneck=bottleneck,
            init_scale=0.005,
            noise_scale=0.0,
        ).to(self.device)
        for gamma in self.steerer.gammas.values():
            gamma.data.fill_(0.02)

        self.manifest = CartridgeManifest(
            cartridge_id=cartridge_id,
            role=role,
            base_model_id=model_name,
            tokenizer_id=model_name,
            steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(self.inject_layers),
            parameter_count=sum(param.numel() for param in self.steerer.parameters()),
            source_corpus=source_corpus,
            metadata={"runtime": "owned-cartridge-harness"},
        )
        self.rack = SteererCartridgeRack()
        self.rack.mount(self.manifest, self.steerer, weight=1.0, active=False)
        self.rack.register_hooks(self.hf_model.model)
        self.enabled = False

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        self.rack.activate(self.manifest.cartridge_id, enabled)

    def set_zero_weights(self, seq_len: int, batch_size: int = 1):
        weights = self.torch.zeros(batch_size, seq_len, 21, device=self.device)
        self.rack.set_weights(weights)

    def generate(self, prompt: str, max_tokens: int = 24) -> str:
        ids = list(self.tokenizer.encode(prompt))
        generated: list[int] = []
        with self.torch.no_grad():
            for _ in range(max_tokens):
                inp = self.torch.tensor([ids[-512:]], device=self.device)
                if self.enabled:
                    self.set_zero_weights(inp.shape[1])
                else:
                    self.rack.activate(self.manifest.cartridge_id, False)
                logits = self.hf_model(inp).logits[0, -1].float()
                if not self.torch.isfinite(logits).all():
                    return "<NONFINITE>"
                next_id = int(logits.argmax())
                ids.append(next_id)
                generated.append(next_id)
                if next_id == self.tokenizer.eos_token_id:
                    break
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def cleanup(self):
        self.rack.remove_hooks()


def train_answer_cartridge(
    runner: QwenAdapterCartridgeRunner,
    train_tasks: list[TaskExample],
    eval_tasks: list[TaskExample],
    out_dir: Path,
    steps: int = 700,
    eval_every: int = 50,
    lr: float = 6e-4,
    seed: int = 23,
) -> dict:
    """Train only the mounted cartridge on prompt/answer rows."""

    torch = runner.torch
    random.seed(seed)
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner.set_enabled(False)
    baseline_rows = evaluate_text_runner(eval_tasks, runner.generate)
    baseline_summary = build_summary(baseline_rows)
    print(
        f"[baseline] {baseline_summary.correct}/{baseline_summary.total} "
        f"acc={baseline_summary.accuracy:.3f}",
        flush=True,
    )

    runner.set_enabled(True)
    runner.steerer.train()
    optimizer = torch.optim.AdamW(runner.steerer.parameters(), lr=lr, weight_decay=0.01)
    best_key = (-1, -1)
    best_state = None
    history: list[dict] = []
    for step in range(1, steps + 1):
        loss = _train_step(runner, train_tasks, optimizer)
        if step % eval_every == 0:
            runner.steerer.eval()
            rows = evaluate_text_runner(eval_tasks, runner.generate)
            summary = build_summary(rows)
            split = summary.by_split
            key = (split.get("heldout", {}).get("correct", 0), summary.correct)
            history.append({"step": step, "loss": loss, **summary.to_json()})
            print(
                f"[eval] step={step} loss={loss:.4f} "
                f"correct={summary.correct}/{summary.total} "
                f"acc={summary.accuracy:.3f} heldout="
                f"{split.get('heldout', {}).get('correct', 0)}/"
                f"{split.get('heldout', {}).get('total', 0)}",
                flush=True,
            )
            if key > best_key:
                best_key = key
                best_state = {
                    state_key: value.detach().cpu().clone()
                    for state_key, value in runner.steerer.state_dict().items()
                }
                torch.save(
                    {
                        "steerer_state": best_state,
                        "manifest": runner.manifest.__dict__,
                        "history": history,
                        "summary": summary.to_json(),
                    },
                    out_dir / "cartridge_best.pt",
                )
            if summary.correct == summary.total:
                print(f"[early_stop] step={step} perfect_eval=1", flush=True)
                break
            runner.steerer.train()

    runner.steerer.eval()
    if best_state is not None:
        runner.steerer.load_state_dict(best_state, strict=False)
    cartridge_rows = evaluate_text_runner(eval_tasks, runner.generate)
    cartridge_summary = build_summary(cartridge_rows)
    print(
        f"[final] {cartridge_summary.correct}/{cartridge_summary.total} "
        f"acc={cartridge_summary.accuracy:.3f}",
        flush=True,
    )
    comparison = compare_rows(baseline_rows, cartridge_rows)
    result = {
        "model": runner.model_name,
        "artifact": str(out_dir / "cartridge_best.pt"),
        "baseline_summary": baseline_summary.to_json(),
        "cartridge_summary": cartridge_summary.to_json(),
        "history": history,
        "baseline_rows": [row.to_json() for row in baseline_rows],
        "cartridge_rows": [row.to_json() for row in cartridge_rows],
        **comparison,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _train_step(runner: QwenAdapterCartridgeRunner, tasks: list[TaskExample], optimizer) -> float:
    import torch.nn.functional as F

    row = random.choice(tasks)
    target_text = f"{row.prompt} {row.expected}\n"
    prompt_ids = runner.tokenizer.encode(row.prompt)
    full_ids = runner.tokenizer.encode(target_text)
    x = runner.torch.tensor([full_ids[:-1]], device=runner.device)
    y = runner.torch.tensor([full_ids[1:]], device=runner.device)
    mask = runner.torch.zeros_like(y, dtype=runner.torch.float32)
    mask[:, max(0, len(prompt_ids) - 1):] = 1.0
    runner.set_zero_weights(x.shape[1])
    logits = runner.hf_model(x).logits.float()
    loss_tokens = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        y.reshape(-1),
        reduction="none",
    ).reshape_as(mask)
    loss = (loss_tokens * mask).sum() / mask.sum().clamp(min=1.0)
    loss = loss + 0.00005 * runner.steerer.orthogonal_penalty()
    optimizer.zero_grad()
    loss.backward()
    runner.torch.nn.utils.clip_grad_norm_(runner.steerer.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def split_tasks(tasks: Iterable[TaskExample]) -> tuple[list[TaskExample], list[TaskExample]]:
    materialized = list(tasks)
    return (
        [task for task in materialized if task.split == "train"],
        materialized,
    )