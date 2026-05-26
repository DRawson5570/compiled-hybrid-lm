#!/usr/bin/env python3
"""Private commonsense cartridge experiment.

This intentionally lives outside hybrid/ while we explore whether a mixed
commonsense curriculum can turn the HellaSwag regression into a gain.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from hybrid.cartridge_harness.qwen import QwenAdapterCartridgeRunner


CARTRIDGE_ID = "qwen-commonsense-mix-cartridge"


@dataclass(frozen=True)
class MCExample:
    source: str
    split: str
    prompt: str
    context: str
    choices: tuple[str, ...]
    answer: int


def _limit(items: list[MCExample], limit: int) -> list[MCExample]:
    return items if limit <= 0 else items[:limit]


def load_hellaswag(split: str, limit: int) -> list[MCExample]:
    from datasets import load_dataset

    raw = load_dataset("hellaswag", split=split)
    out: list[MCExample] = []
    for item in raw:
        endings = tuple(str(x) for x in item["endings"][:4])
        label = item.get("label")
        if isinstance(label, str):
            label = int(label)
        ctx = str(item["ctx"])
        out.append(MCExample("hellaswag", split, ctx, ctx, endings, int(label)))
        if limit > 0 and len(out) >= limit:
            break
    return out


def load_piqa(split: str, limit: int) -> list[MCExample]:
    from datasets import load_dataset

    raw = load_dataset("piqa", split=split)
    out: list[MCExample] = []
    for item in raw:
        goal = str(item["goal"])
        choices = (str(item["sol1"]), str(item["sol2"]))
        prompt = f"Goal: {goal}\nWhich solution is more physically plausible?"
        out.append(MCExample("piqa", split, prompt, prompt, choices, int(item["label"])))
        if limit > 0 and len(out) >= limit:
            break
    return out


def load_commonsenseqa(split: str, limit: int) -> list[MCExample]:
    from datasets import load_dataset

    raw = load_dataset("commonsense_qa", split=split)
    out: list[MCExample] = []
    for item in raw:
        labels = list(item["choices"]["label"])
        texts = tuple(str(x) for x in item["choices"]["text"])
        answer_key = item.get("answerKey")
        if answer_key not in labels:
            continue
        prompt = f"Question: {item['question']}\nWhich answer is most commonsense?"
        out.append(MCExample("commonsenseqa", split, prompt, prompt, texts, labels.index(answer_key)))
        if limit > 0 and len(out) >= limit:
            break
    return out


def load_openbookqa(split: str, limit: int) -> list[MCExample]:
    from datasets import load_dataset

    raw = load_dataset("openbookqa", split=split)
    out: list[MCExample] = []
    for item in raw:
        labels = list(item["choices"]["label"])
        texts = tuple(str(x) for x in item["choices"]["text"])
        answer_key = item.get("answerKey")
        if answer_key not in labels:
            continue
        prompt = f"Question: {item['question_stem']}\nWhich answer is best supported by everyday science?"
        out.append(MCExample("openbookqa", split, prompt, prompt, texts, labels.index(answer_key)))
        if limit > 0 and len(out) >= limit:
            break
    return out


def add_dataset(target: list[MCExample], label: str, loader, split: str, limit: int):
    if limit == 0:
        return
    try:
        rows = loader(split, limit)
    except Exception as exc:
        print(f"[data] skip {label}/{split}: {type(exc).__name__}: {exc}", flush=True)
        return
    target += rows
    print(f"[data] loaded {label}/{split}: {len(rows)}", flush=True)


def score_choice(runner: QwenAdapterCartridgeRunner, ex: MCExample, choice: str, enabled: bool) -> torch.Tensor:
    full_text = f"{ex.context} {choice}"
    full_ids = runner.tokenizer.encode(full_text, return_tensors="pt").to(runner.device)
    ctx_ids = runner.tokenizer.encode(ex.context, return_tensors="pt")
    answer_len = full_ids.shape[1] - ctx_ids.shape[1]
    if answer_len <= 0:
        return torch.tensor(float("-inf"), device=runner.device)
    runner.set_enabled(enabled)
    if enabled:
        runner.set_zero_weights(full_ids.shape[1])
    logits = runner.hf_model(full_ids).logits.float()
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    total = torch.tensor(0.0, device=runner.device)
    for j in range(answer_len):
        pos = ctx_ids.shape[1] + j - 1
        token_id = int(full_ids[0, pos + 1].item())
        total = total + logprobs[0, pos, token_id]
    return total / max(answer_len, 1)


def accuracy(runner: QwenAdapterCartridgeRunner, examples: list[MCExample], enabled: bool) -> dict:
    correct = 0
    by_source: dict[str, dict[str, int]] = {}
    runner.steerer.eval()
    with torch.no_grad():
        for ex in examples:
            scores = [float(score_choice(runner, ex, choice, enabled).detach().cpu()) for choice in ex.choices]
            pred = max(range(len(scores)), key=lambda idx: scores[idx])
            hit = int(pred == ex.answer)
            correct += hit
            bucket = by_source.setdefault(ex.source, {"correct": 0, "total": 0})
            bucket["correct"] += hit
            bucket["total"] += 1
    return {
        "accuracy": correct / max(len(examples), 1),
        "correct": correct,
        "total": len(examples),
        "by_source": {k: {**v, "accuracy": v["correct"] / max(v["total"], 1)} for k, v in by_source.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bottleneck", type=int, default=96)
    parser.add_argument("--hellaswag-train", type=int, default=5000)
    parser.add_argument("--piqa-train", type=int, default=2000)
    parser.add_argument("--commonsenseqa-train", type=int, default=1200)
    parser.add_argument("--openbookqa-train", type=int, default=800)
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = []
    add_dataset(train, "hellaswag", load_hellaswag, "train", args.hellaswag_train)
    add_dataset(train, "piqa", load_piqa, "train", args.piqa_train)
    add_dataset(train, "commonsenseqa", load_commonsenseqa, "train", args.commonsenseqa_train)
    add_dataset(train, "openbookqa", load_openbookqa, "train", args.openbookqa_train)
    random.shuffle(train)
    eval_set = []
    add_dataset(eval_set, "hellaswag", load_hellaswag, "validation", args.eval_limit)
    add_dataset(eval_set, "piqa", load_piqa, "validation", args.eval_limit)
    add_dataset(eval_set, "commonsenseqa", load_commonsenseqa, "validation", args.eval_limit)
    add_dataset(eval_set, "openbookqa", load_openbookqa, "validation", args.eval_limit)
    if not train or not eval_set:
        raise RuntimeError(f"empty experiment data: train={len(train)} eval={len(eval_set)}")

    runner = QwenAdapterCartridgeRunner(
        args.model,
        device=args.device,
        bottleneck=args.bottleneck,
        cartridge_id=CARTRIDGE_ID,
        source_corpus="hellaswag+piqa+commonsenseqa",
    )
    opt = torch.optim.AdamW(runner.steerer.parameters(), lr=args.lr, weight_decay=0.01)
    history: list[dict] = []
    best = {"accuracy": -1.0}
    best_state = None

    config = vars(args) | {
        "cartridge_id": CARTRIDGE_ID,
        "train_count": len(train),
        "eval_count": len(eval_set),
        "train_by_source": {src: sum(1 for ex in train if ex.source == src) for src in sorted({ex.source for ex in train})},
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    raw = accuracy(runner, eval_set, enabled=False)
    print(f"[baseline] raw={raw['accuracy']:.4f} by_source={raw['by_source']}", flush=True)
    (out_dir / "raw_eval.json").write_text(json.dumps(raw, indent=2) + "\n")

    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        runner.steerer.train()
        ex = random.choice(train)
        scores = torch.stack([score_choice(runner, ex, choice, enabled=True) for choice in ex.choices])
        loss = torch.nn.functional.cross_entropy(
            scores.unsqueeze(0), torch.tensor([ex.answer], device=runner.device)
        )
        loss = loss + 0.00005 * runner.steerer.orthogonal_penalty()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(runner.steerer.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = accuracy(runner, eval_set, enabled=True)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "elapsed_sec": time.perf_counter() - t0,
                "cartridge_eval": val,
                "delta_vs_raw": val["accuracy"] - raw["accuracy"],
            }
            history.append(row)
            with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"[train] step={step} loss={row['loss']:.4f} acc={val['accuracy']:.4f} "
                f"delta={row['delta_vs_raw']:+.4f} by_source={val['by_source']}",
                flush=True,
            )
            if val["accuracy"] > best["accuracy"]:
                best = val
                best_state = {k: v.detach().cpu().clone() for k, v in runner.steerer.state_dict().items()}
                torch.save(
                    {
                        "steerer_state": best_state,
                        "manifest": runner.manifest.__dict__,
                        "raw_eval": raw,
                        "best_eval": best,
                        "history": history,
                    },
                    out_dir / "cartridge_best.pt",
                )

    summary = {"raw_eval": raw, "best_eval": best, "history": history}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    runner.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())