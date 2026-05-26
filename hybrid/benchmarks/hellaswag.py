"""HellaSwag commonsense benchmark harness.

Shares the same evaluation patterns as ARC but with HellaSwag-specific data format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class HellaSwagExample:
    id: str
    ctx: str
    endings: list[str]
    label: int | None
    split: str


def load_hellaswag_dataset(
    dataset_name: str = "Rowan/hellaswag",
    split: str = "validation",
    local_jsonl: str | None = None,
    max_examples: int = 0,
) -> list[HellaSwagExample]:
    if local_jsonl:
        with open(local_jsonl, encoding="utf-8") as fh:
            raw = [json.loads(line) for line in fh if line.strip()]
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, trust_remote_code=True)
        raw = [dict(item) for item in ds[split]]

    examples = []
    for i, item in enumerate(raw):
        label = item.get("label")
        if isinstance(label, str) and label.isdigit():
            label = int(label)
        elif isinstance(label, (int, float)):
            label = int(label)
        ex = HellaSwagExample(
            id=str(item.get("ind", item.get("id", i))),
            ctx=item.get("ctx", ""),
            endings=item.get("endings", item.get("choices", [])),
            label=label,
            split=split,
        )
        if not ex.ctx.strip() or len(ex.endings) < 2:
            continue
        examples.append(ex)

    if max_examples > 0:
        examples = examples[:max_examples]
    return examples


TEMPLATE_SOURCE = (
    "Context: {ctx}\n\n"
    "Which ending is most plausible?\n"
    "A. {ending_0}\n"
    "B. {ending_1}\n"
    "C. {ending_2}\n"
    "D. {ending_3}\n\n"
    "Answer:"
)
TEMPLATE_HASH = hashlib.sha256(TEMPLATE_SOURCE.encode()).hexdigest()


def render_prompt(example: HellaSwagExample) -> str:
    endings = example.endings[:4]
    while len(endings) < 4:
        endings.append("")
    return TEMPLATE_SOURCE.format(
        ctx=example.ctx,
        ending_0=endings[0],
        ending_1=endings[1],
        ending_2=endings[2],
        ending_3=endings[3],
    )


def render_continuation(ending_text: str) -> str:
    return f" {ending_text}"


def score_ending_logprob(
    model, tokenizer, ctx: str, ending: str, device: torch.device
) -> tuple[float, float, int]:
    full_text = f"{ctx} {ending}"
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    ctx_ids = tokenizer.encode(ctx, return_tensors="pt")
    answer_len = full_ids.shape[1] - ctx_ids.shape[1]
    if answer_len <= 0:
        return float("-inf"), float("-inf"), 0

    with torch.no_grad():
        logits = model(full_ids).logits.float()
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    for j in range(answer_len):
        pos = ctx_ids.shape[1] + j - 1
        token_id = full_ids[0, pos + 1].item()
        total_logprob += logprobs[0, pos, token_id].item()

    score_norm = total_logprob / max(answer_len, 1)
    return score_norm, total_logprob, answer_len


def run_hellaswag_baseline(
    examples: list[HellaSwagExample],
    model_name: str,
    device: str,
    report_dir: Path | None = None,
    log_every: int = 100,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch_device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(torch_device)
    model.eval()

    predictions = []
    t_start = time.perf_counter()
    correct_norm = correct_sum = 0
    total = 0

    for i, ex in enumerate(examples):
        prompt = render_prompt(ex)
        scores = []
        for ending in ex.endings[:4]:
            norm, total_score, n_tokens = score_ending_logprob(model, tokenizer, ex.ctx, ending, torch_device)
            scores.append({"norm": norm, "sum": total_score, "num_tokens": n_tokens})

        pred_norm = max(range(len(scores)), key=lambda j: scores[j]["norm"])
        pred_sum = max(range(len(scores)), key=lambda j: scores[j]["sum"])
        correct_n = (pred_norm == ex.label) if ex.label is not None else None
        correct_s = (pred_sum == ex.label) if ex.label is not None else None
        if correct_n:
            correct_norm += 1
        if correct_s:
            correct_sum += 1
        total += 1

        predictions.append({
            "id": ex.id,
            "ctx": ex.ctx,
            "endings": ex.endings[:4],
            "label": ex.label,
            "scores_norm": [s["norm"] for s in scores],
            "scores_sum": [s["sum"] for s in scores],
            "pred_norm": pred_norm,
            "pred_sum": pred_sum,
            "correct_norm": correct_n,
            "correct_sum": correct_s,
        })

        if (i + 1) % log_every == 0 or i == len(examples) - 1:
            print(f"  [{i + 1}/{len(examples)}] acc_norm={correct_norm / total:.4f} "
                  f"({(time.perf_counter() - t_start):.1f}s)", flush=True)

    acc_norm = correct_norm / max(total, 1)
    acc_sum = correct_sum / max(total, 1)
    result = {
        "benchmark": "hellaswag",
        "dataset": "Rowan/hellaswag",
        "split": examples[0].split if examples else "validation",
        "model": model_name,
        "mode": "hf-causal",
        "prompt_template_sha256": TEMPLATE_HASH,
        "accuracy_norm": acc_norm,
        "accuracy_sum": acc_sum,
        "total": total,
        "correct_norm": correct_norm,
        "correct_sum": correct_sum,
        "duration_sec": time.perf_counter() - t_start,
    }

    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        with open(report_dir / "predictions.jsonl", "w") as fh:
            for p in predictions:
                fh.write(json.dumps(p) + "\n")

    print(f"\nFinal: accuracy_norm = {correct_norm}/{total} = {acc_norm:.4f}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="HellaSwag benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--dataset", default="Rowan/hellaswag")
    eval_p.add_argument("--split", default="validation")
    eval_p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--max-examples", type=int, default=0)
    eval_p.add_argument("--report-dir", default=None)
    eval_p.add_argument("--log-every", type=int, default=100)

    args = parser.parse_args()

    examples = load_hellaswag_dataset(
        dataset_name=args.dataset,
        split=args.split,
        max_examples=args.max_examples,
    )
    print(f"Loaded {len(examples)} examples from {args.dataset}/{args.split}", flush=True)

    report_dir = Path(args.report_dir) if args.report_dir else None
    run_hellaswag_baseline(examples, args.model, args.device, report_dir, args.log_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
