"""MMLU benchmark harness — broad cartridge with multiple-choice logprob scoring."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch


MMLU_CARTRIDGE_ID = "qwen-mmlu-general-cartridge"

SUBJECT_FAMILIES = {
    "stem": ["abstract_algebra","anatomy","astronomy","college_biology","college_chemistry",
             "college_computer_science","college_mathematics","college_medicine","college_physics",
             "computer_security","conceptual_physics","electrical_engineering","elementary_mathematics",
             "formal_logic","high_school_biology","high_school_chemistry","high_school_computer_science",
             "high_school_mathematics","high_school_physics","high_school_statistics","machine_learning",
             "medical_genetics","virology"],
    "humanities": ["college_european_history","high_school_european_history","high_school_us_history",
                   "high_school_world_history","international_law","jurisprudence","logical_fallacies",
                   "moral_disputes","moral_scenarios","philosophy","prehistory","professional_law",
                   "world_religions"],
    "social_sciences": ["business_ethics","clinical_knowledge","econometrics","global_facts",
                        "high_school_geography","high_school_government_and_politics",
                        "high_school_macroeconomics","high_school_microeconomics",
                        "high_school_psychology","human_aging","human_sexuality",
                        "management","marketing","miscellaneous","nutrition",
                        "professional_accounting","professional_medicine",
                        "professional_psychology","public_relations","security_studies",
                        "sociology","us_foreign_policy"],
}

def _subject_family(subject: str) -> str:
    for family, subjects in SUBJECT_FAMILIES.items():
        if subject in subjects:
            return family
    return "other"


@dataclass
class MMLUExample:
    id: str
    question: str
    choices: list[str]
    answer: int | None
    subject: str
    split: str


def load_mmlu_dataset(
    dataset_name: str = "cais/mmlu",
    split: str = "test",
    max_per_subject: int = 0,
    max_examples: int = 0,
) -> list[MMLUExample]:
    from datasets import load_dataset
    ds = load_dataset(dataset_name, "all", trust_remote_code=True)
    raw = list(ds[split])

    examples = []
    counts = {}
    for item in raw:
        subject = item["subject"]
        if max_per_subject > 0 and counts.get(subject, 0) >= max_per_subject:
            continue
        example = MMLUExample(
            id=f"{subject}_{item['question'][:40]}",
            question=item["question"],
            choices=item["choices"],
            answer=item.get("answer"),
            subject=subject,
            split=split,
        )
        examples.append(example)
        counts[subject] = counts.get(subject, 0) + 1

    if max_examples > 0:
        examples = examples[:max_examples]
    return examples


def render_mmlu_prompt(example: MMLUExample) -> str:
    labels = [chr(ord('A') + i) for i in range(len(example.choices))]
    choices_block = "\n".join(f"{l}. {c}" for l, c in zip(labels, example.choices))
    return f"Question: {example.question}\n\nChoices:\n{choices_block}\n\nAnswer:"


def score_choice_logprob(model, tokenizer, prompt: str, choice: str, device) -> tuple[float, float, int]:
    full_text = prompt + " " + choice
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
    answer_len = full_ids.shape[1] - prompt_ids.shape[1]
    if answer_len <= 0:
        return float("-inf"), float("-inf"), 0

    with torch.no_grad():
        logits = model(full_ids).logits.float()
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

    total = 0.0
    for j in range(answer_len):
        pos = prompt_ids.shape[1] + j - 1
        token_id = full_ids[0, pos + 1].item()
        total += logprobs[0, pos, token_id].item()

    return total / max(answer_len, 1), total, answer_len


def run_mmlu_baseline(
    examples: list[MMLUExample],
    model_name: str,
    device: str,
    report_dir: Path | None = None,
    log_every: int = 250,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if torch_device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(torch_device)
    model.eval()

    correct_norm = 0
    total = 0
    t0 = time.perf_counter()
    per_family = {}
    predictions = []

    for ex in examples:
        prompt = render_mmlu_prompt(ex)
        scores = []
        for choice in ex.choices:
            norm, total_score, nt = score_choice_logprob(model, tokenizer, prompt, choice, torch_device)
            scores.append({"norm": norm, "sum": total_score, "num_tokens": nt})

        if ex.answer is not None:
            pred = max(range(len(scores)), key=lambda j: scores[j]["norm"])
            correct = pred == ex.answer
            if correct:
                correct_norm += 1
            family = _subject_family(ex.subject)
            pf = per_family.setdefault(family, {"correct": 0, "total": 0})
            pf["correct"] += int(correct)
            pf["total"] += 1
            total += 1

        predictions.append({
            "id": ex.id, "subject": ex.subject, "answer": ex.answer,
            "scores_norm": [s["norm"] for s in scores],
            "pred_norm": pred if ex.answer is not None else None,
            "correct_norm": correct if ex.answer is not None else None,
        })

        if total and total % log_every == 0:
            print(f"  [{total}] acc={correct_norm/total:.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)

    acc = correct_norm / max(total, 1)
    result = {"benchmark": "mmlu", "model": model_name, "accuracy_norm": acc,
              "total": total, "correct_norm": correct_norm,
              "per_family": {k: v["correct"]/max(v["total"],1) for k,v in per_family.items()},
              "per_family_counts": {k: v["total"] for k,v in per_family.items()},
              "duration_sec": time.perf_counter()-t0}

    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
        with open(report_dir / "predictions.jsonl", "w") as f:
            for p in predictions:
                f.write(json.dumps(p)+"\n")

    print(f"Final: accuracy={acc:.4f}, per_family={result['per_family']}", flush=True)
    return result


def train_mlu_cartridge(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    device: str = "cuda",
    out_dir: str = "",
    steps: int = 500,
    lr: float = 2e-4,
    eval_every: int = 50,
    train_max_examples: int = 2000,
    train_max_per_subject: int = 50,
    bottleneck: int = 64,
    seed: int = 23,
) -> dict:
    from hybrid.cartridge_harness.qwen import QwenAdapterCartridgeRunner

    random.seed(seed)
    train_ex = load_mmlu_dataset(split="auxiliary_train", max_per_subject=train_max_per_subject, max_examples=train_max_examples)
    val_ex = load_mmlu_dataset(split="validation")
    print(f"Train: {len(train_ex)} ({len(set(e.subject for e in train_ex))} subjects), Val: {len(val_ex)}")

    runner = QwenAdapterCartridgeRunner(model_name, device=device, bottleneck=bottleneck,
                                         cartridge_id=MMLU_CARTRIDGE_ID)
    runner.set_enabled(True)
    runner.steerer.train()
    opt = runner.torch.optim.AdamW(runner.steerer.parameters(), lr=lr, weight_decay=0.01)
    best_acc = -1.0; best_state = None
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    for step in range(1, steps+1):
        ex = random.choice(train_ex)
        prompt = render_mmlu_prompt(ex)

        scores = []
        for choice in ex.choices:
            full_text = prompt + " " + choice
            full_ids = runner.tokenizer.encode(full_text, return_tensors="pt").to(runner.device)
            prompt_ids = runner.tokenizer.encode(prompt, return_tensors="pt")
            answer_len = full_ids.shape[1] - prompt_ids.shape[1]
            if answer_len <= 0:
                scores.append(runner.torch.tensor(float("-inf"), device=runner.device, requires_grad=True))
                continue
            runner.set_zero_weights(full_ids.shape[1])
            logits = runner.hf_model(full_ids).logits.float()
            logprobs = runner.torch.nn.functional.log_softmax(logits, dim=-1)
            total_logprob = runner.torch.tensor(0.0, device=runner.device, requires_grad=True)
            for j in range(answer_len):
                pos = prompt_ids.shape[1] + j - 1
                token_id = full_ids[0, pos+1].item()
                total_logprob = total_logprob + logprobs[0, pos, token_id]
            scores.append(total_logprob / max(answer_len, 1))

        correct_idx = ex.answer if ex.answer is not None else 0
        scores_tensor = runner.torch.stack(scores)
        loss = runner.torch.nn.functional.cross_entropy(
            scores_tensor.unsqueeze(0), runner.torch.tensor([correct_idx], device=runner.device))
        loss = loss + 0.00005 * runner.steerer.orthogonal_penalty()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        runner.torch.nn.utils.clip_grad_norm_(runner.steerer.parameters(), 1.0)
        opt.step()

        if step == 1 or step % eval_every == 0 or step == steps:
            runner.steerer.eval()
            val_acc = _mmlu_accuracy(runner, val_ex[:50])
            print(f"[mmlu-train] step={step} loss={float(loss.detach().cpu()):.4f} val_acc={val_acc:.4f}", flush=True)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k,v in runner.steerer.state_dict().items()}
                runner.torch.save(
                    {"steerer_state": best_state, "manifest": runner.manifest.__dict__,
                     "val_accuracy": val_acc},
                    out_path / "cartridge_best.pt")
            runner.steerer.train()

    if best_state:
        runner.steerer.load_state_dict(best_state, strict=False)
    (out_path / "train_config.json").write_text(json.dumps({
        "cartridge_id": MMLU_CARTRIDGE_ID, "train_count": len(train_ex),
        "steps": steps, "best_val_accuracy": best_acc,
    }, indent=2)+"\n")
    runner.cleanup()
    return {"best_val_accuracy": best_acc, "artifact": str(out_path / "cartridge_best.pt")}


def _mmlu_accuracy(runner, examples, max_examples=50):
    correct = 0; total = 0
    for ex in examples[:max_examples]:
        if ex.answer is None: continue
        prompt = render_mmlu_prompt(ex)
        scores = []
        for choice in ex.choices:
            norm, _, _ = score_choice_logprob(runner.hf_model, runner.tokenizer, prompt, choice, runner.device)
            scores.append(norm)
        pred = max(range(len(scores)), key=lambda j: scores[j])
        total += 1
        if pred == ex.answer: correct += 1
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="MMLU benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--split", default="test")
    eval_p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    eval_p.add_argument("--device", default="cuda")
    eval_p.add_argument("--max-examples", type=int, default=0)
    eval_p.add_argument("--report-dir", default=None)

    train_p = sub.add_parser("train-cartridge")
    train_p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--out-dir", required=True)
    train_p.add_argument("--steps", type=int, default=500)
    train_p.add_argument("--lr", type=float, default=2e-4)
    train_p.add_argument("--eval-every", type=int, default=50)
    train_p.add_argument("--train-max-examples", type=int, default=2000)
    train_p.add_argument("--train-max-per-subject", type=int, default=50)

    args = parser.parse_args()
    if args.command == "eval":
        examples = load_mmlu_dataset(split=args.split, max_examples=args.max_examples)
        print(f"Loaded {len(examples)} MMLU examples", flush=True)
        run_mmlu_baseline(examples, args.model, args.device, Path(args.report_dir) if args.report_dir else None)
    elif args.command == "train-cartridge":
        train_mlu_cartridge(model_name=args.model, device=args.device, out_dir=args.out_dir,
                             steps=args.steps, lr=args.lr, eval_every=args.eval_every,
                             train_max_examples=args.train_max_examples,
                             train_max_per_subject=args.train_max_per_subject)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
