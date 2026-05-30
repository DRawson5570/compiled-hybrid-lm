"""ARC cartridge training with option-ranking cross-entropy loss.

Trains only the mounted adapter cartridge; Qwen base remains frozen.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

from hybrid.benchmarks.arc_data import ARCExample
from hybrid.benchmarks.arc_prompts import PromptTemplate, get_template
from hybrid.benchmarks.arc_reports import write_reports
from hybrid.benchmarks.arc_scoring import HFArcScorer
from hybrid.cartridge_harness.qwen import QwenAdapterCartridgeRunner


def _option_logprob(
    runner: QwenAdapterCartridgeRunner,
    prompt: str,
    continuation: str,
) -> tuple[float, float, int]:
    prompt_ids = runner.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    full_text = prompt + continuation
    full_ids = runner.tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
    full_ids = full_ids.to(runner.device)
    answer_ids = full_ids[0, prompt_ids.shape[1]:]
    if answer_ids.numel() == 0:
        continuation_space = " " + continuation
        full_text = prompt + continuation_space
        full_ids = runner.tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
        full_ids = full_ids.to(runner.device)
        answer_ids = full_ids[0, prompt_ids.shape[1]:]
    if answer_ids.numel() == 0:
        return float("-inf"), float("-inf"), 0

    if not runner.enabled:
        runner.rack.activate(runner.manifest.cartridge_id, True)
        was_inactive = True
    else:
        was_inactive = False

    runner.set_zero_weights(full_ids.shape[1])

    logits = runner.hf_model(full_ids).logits.float()
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    if was_inactive:
        runner.rack.activate(runner.manifest.cartridge_id, False)

    total_logprob = 0.0
    for i, token_id in enumerate(answer_ids):
        pos = prompt_ids.shape[1] + i - 1
        token_logprob = logprobs[0, pos, token_id].item()
        total_logprob += token_logprob

    num_tokens = answer_ids.numel()
    score_norm = total_logprob / max(num_tokens, 1)
    score_sum = total_logprob
    return score_norm, score_sum, num_tokens


def _option_scores_with_grad(
    runner: QwenAdapterCartridgeRunner,
    prompt: str,
    continuations: list[str],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute option scores with gradients flowing through adapter."""
    full_ids_list = []
    answer_idxs_list = []

    for continuation in continuations:
        prompt_ids = runner.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        full_text = prompt + continuation
        full_ids = runner.tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
        full_ids = full_ids.to(runner.device)
        answer_ids = full_ids[0, prompt_ids.shape[1]:]
        if answer_ids.numel() == 0:
            continuation_space = " " + continuation
            full_text = prompt + continuation_space
            full_ids = runner.tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")
            full_ids = full_ids.to(runner.device)
            answer_ids = full_ids[0, prompt_ids.shape[1]:]
        if answer_ids.numel() == 0:
            return torch.tensor([float("-inf")] * len(continuations), device=runner.device)
        full_ids_list.append(full_ids)
        answer_idxs_list.append((prompt_ids.shape[1], answer_ids))

    assert len(full_ids_list) == len(continuations)

    scores = []
    for i, full_ids in enumerate(full_ids_list):
        prompt_len, answer_ids = answer_idxs_list[i]
        runner.set_zero_weights(full_ids.shape[1])
        logits = runner.hf_model(full_ids).logits.float()
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        total_logprob = torch.tensor(0.0, device=runner.device, requires_grad=True)
        for j, token_id in enumerate(answer_ids):
            pos = prompt_len + j - 1
            token_logprob = logprobs[0, pos, token_id]
            total_logprob = total_logprob + token_logprob

        score_norm = total_logprob / max(answer_ids.numel(), 1)
        scores.append(score_norm)

    return torch.stack(scores) / temperature


def train_arc_cartridge(
    runner: QwenAdapterCartridgeRunner,
    train_examples: list[ARCExample],
    val_examples: list[ARCExample],
    out_dir: Path,
    template_id: str = "arc_v1",
    steps: int = 500,
    lr: float = 2e-4,
    eval_every: int = 50,
    temperature: float = 1.0,
    lambda_margin: float = 0.0,
    seed: int = 23,
) -> dict:
    random.seed(seed)
    runner.torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    template = get_template(template_id)

    runner.set_enabled(True)
    runner.steerer.train()
    optimizer = runner.torch.optim.AdamW(runner.steerer.parameters(), lr=lr, weight_decay=0.01)

    best_accuracy = -1.0
    best_state = None
    history: list[dict] = []

    for step in range(1, steps + 1):
        example = random.choice(train_examples)
        correct_idx = next(
            (i for i, c in enumerate(example.choices) if c.label == example.answer_key),
            0,
        )

        prompt = template.render_prompt(example)
        continuations = [template.render_continuation(c.text) for c in example.choices]

        scores = _option_scores_with_grad(runner, prompt, continuations, temperature)
        loss = torch.nn.functional.cross_entropy(
            scores.unsqueeze(0),
            torch.tensor([correct_idx], device=runner.device),
        )

        if lambda_margin > 0:
            correct_score = scores[correct_idx]
            for wrong_idx in range(len(scores)):
                if wrong_idx != correct_idx:
                    loss = loss + lambda_margin * torch.clamp(
                        0.0 - (correct_score - scores[wrong_idx]), min=0.0
                    )

        loss = loss + 0.00005 * runner.steerer.orthogonal_penalty()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        runner.torch.nn.utils.clip_grad_norm_(runner.steerer.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            runner.steerer.eval()
            val_acc = _eval_accuracy(runner, val_examples, template)
            history.append({"step": step, "loss": float(loss.detach().cpu()), "val_accuracy": val_acc})
            print(
                f"[arc-train] step={step} loss={float(loss.detach().cpu()):.4f} "
                f"val_acc={val_acc:.4f}",
                flush=True,
            )
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in runner.steerer.state_dict().items()
                }
                runner.torch.save(
                    {
                        "steerer_state": best_state,
                        "manifest": runner.manifest.__dict__,
                        "history": history,
                        "val_accuracy": val_acc,
                    },
                    out_dir / "cartridge_best.pt",
                )
            runner.steerer.train()

    runner.steerer.eval()
    if best_state is not None:
        runner.steerer.load_state_dict(best_state, strict=False)

    (out_dir / "train_config.json").write_text(json.dumps({
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "steps": steps,
        "lr": lr,
        "eval_every": eval_every,
        "temperature": temperature,
        "lambda_margin": lambda_margin,
        "prompt_template": template_id,
        "prompt_template_sha256": template.hash(),
        "seed": seed,
    }, indent=2), encoding="utf-8")

    metrics = []
    for entry in history:
        metrics.append(json.dumps(entry))
    (out_dir / "metrics.jsonl").write_text("\n".join(metrics) + "\n", encoding="utf-8")

    runner.set_enabled(True)
    runner.steerer.eval()
    val_scorer = HFArcScorer(runner.hf_model, runner.tokenizer, runner.device)
    scored = []
    for example in val_examples:
        se = val_scorer.score_example(example, template)
        scored.append(se)

    if scored:
        correct = sum(1 for s in scored if s.correct_norm)
        total = sum(1 for s in scored if s.example.answer_key is not None)
        final_acc = correct / max(total, 1)
        print(f"[arc-train] final val accuracy: {correct}/{total} = {final_acc:.4f}", flush=True)

        _ = write_reports(
            out_dir / "validation_report",
            scored,
            0,
            {
                "config": val_examples[0].config,
                "dataset": "allenai/ai2_arc",
                "split": val_examples[0].split,
                "model": runner.model_name,
                "mode": "qwen-single-cartridge",
                "prompt_template": template_id,
                "prompt_template_sha256": template.hash(),
                "duration_sec": 0.0,
                "started_at": "",
            },
        )
    else:
        final_acc = best_accuracy

    result = {
        "model": runner.model_name,
        "artifact": str(out_dir / "cartridge_best.pt"),
        "best_val_accuracy": best_accuracy,
        "final_val_accuracy": final_acc,
        "history": history,
        "train_count": len(train_examples),
        "val_count": len(val_examples),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _eval_accuracy(
    runner: QwenAdapterCartridgeRunner,
    examples: list[ARCExample],
    template: PromptTemplate,
    max_examples: int = 50,
) -> float:
    correct = 0
    total = 0
    sample = examples[:max_examples]
    tokenizer = runner.tokenizer
    device = runner.device

    for example in sample:
        if example.answer_key is None:
            continue
        prompt = template.render_prompt(example)
        option_scores = []
        for choice in example.choices:
            continuation = template.render_continuation(choice.text)
            norm, _, _ = _option_logprob(runner, prompt, continuation)
            option_scores.append((choice.label, norm))

        if all(math.isinf(s) for _, s in option_scores):
            continue
        best_label, _ = max(option_scores, key=lambda x: x[1])
        total += 1
        if best_label == example.answer_key:
            correct += 1

    return correct / max(total, 1)
