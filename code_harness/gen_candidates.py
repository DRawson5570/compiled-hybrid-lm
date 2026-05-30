#!/usr/bin/env python3
"""Generate + execute candidate completions for execution-feedback training.

For each problem we sample N completions from the FROZEN base model (batched),
run the real unit tests, and cache {task_id, prompt_text, candidates:[{code,passed}]}.

Supports two datasets:
  - mbpp     : google-research-datasets/mbpp 'sanitized' (real assert test_list)
  - humaneval: openai_humaneval (held-out eval; for reranking eval cache)

The cache is reused by train_rerank.py / train_rft.py so we never pay for the
(slow on M40) sampling more than once.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL = "Qwen/Qwen3.5-4B"

MBPP_INSTRUCTION = (
    "{nl}\n\nYour code must pass these tests:\n{tests}\n\n"
    "Return ONLY the function inside a single ```python code block, no explanation."
)
HE_INSTRUCTION = (
    "Complete the following Python function. Return ONLY the complete function "
    "(including the signature) inside a single ```python code block, no explanation.\n\n"
    "```python\n{prompt}\n```"
)


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def run_program(program: str, timeout: float = 8.0) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
        tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def mbpp_problems(split):
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    out = []
    for ex in ds:
        out.append({
            "task_id": f"mbpp/{ex['task_id']}",
            "nl": ex["prompt"].strip(),
            "tests": ex["test_list"],
            "test_imports": ex.get("test_imports", []) or [],
            "entry_point": None,
        })
    return out


def humaneval_problems(split="test"):
    ds = load_dataset("openai_humaneval", split=split)
    out = []
    for ex in ds:
        out.append({
            "task_id": ex["task_id"],
            "prompt": ex["prompt"],
            "test": ex["test"],
            "entry_point": ex["entry_point"],
        })
    return out


def build_mbpp_program(code, prob):
    code = code.strip("\n")
    setup = "\n".join(prob["test_imports"])
    tests = "\n".join(prob["tests"])
    return f"{setup}\n{code}\n{tests}\n"


def build_he_program(code, prob):
    code = code.strip("\n")
    if f"def {prob['entry_point']}" in code:
        prog = code
    else:
        prog = f"{prob['prompt']}{code}"
    return f"{prog}\n\n{prob['test']}\n\ncheck({prob['entry_point']})\n"


def chat_text(tok, content):
    msgs = [{"role": "user", "content": content}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mbpp", "humaneval"], required=True)
    ap.add_argument("--split", default=None, help="mbpp: train/test/validation; he: test")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=0, help="0 = to end")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map={"": args.device}, trust_remote_code=True).eval()

    if args.dataset == "mbpp":
        split = args.split or "train"
        probs = mbpp_problems(split)
        instr = lambda p: MBPP_INSTRUCTION.format(nl=p["nl"], tests="\n".join(p["tests"]))
        build = build_mbpp_program
    else:
        probs = humaneval_problems(args.split or "test")
        instr = lambda p: HE_INSTRUCTION.format(prompt=p["prompt"])
        build = build_he_program

    if args.limit:
        probs = probs[: args.limit]
    if args.shard_end or args.shard_start:
        end = args.shard_end or len(probs)
        probs = probs[args.shard_start:end]
        print(f"shard [{args.shard_start}:{end}] -> {len(probs)} problems", flush=True)

    # Build the full sampling job: each problem repeated n_samples times.
    jobs = []  # (prob_idx, prompt_text)
    for pi, p in enumerate(probs):
        text = chat_text(tok, instr(p))
        for _ in range(args.n_samples):
            jobs.append((pi, text))

    results = {pi: [] for pi in range(len(probs))}
    import time
    t0 = time.time()
    done = 0
    for b in range(0, len(jobs), args.batch):
        batch = jobs[b: b + args.batch]
        texts = [t for _, t in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, top_p=0.95, pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for (pi, _), text in zip(batch, decoded):
            code = extract_code(text)
            program = build(code, probs[pi])
            passed = run_program(program)
            results[pi].append({"code": code, "passed": passed})
        done += len(batch)
        rate = done / max(time.time() - t0, 1e-6)
        print(f"[{done}/{len(jobs)}] {rate:.1f} gen/s", flush=True)

    # Write cache
    n_pass_total = 0
    n_usable = 0
    with open(args.out, "w") as f:
        for pi, p in enumerate(probs):
            cands = results[pi]
            np_ = sum(c["passed"] for c in cands)
            n_pass_total += np_
            rec = {"task_id": p["task_id"], "candidates": cands}
            rec.update({k: p[k] for k in p if k not in ("nl",)})
            if args.dataset == "mbpp":
                rec["nl"] = p["nl"]
            if 0 < np_ < len(cands):
                n_usable += 1
            f.write(json.dumps(rec) + "\n")

    npb = len(jobs)
    print(f"\nWrote {len(probs)} problems, {npb} candidates to {args.out}")
    print(f"pass rate (candidate-level pass@1-ish): {n_pass_total}/{npb} = {n_pass_total/npb:.1%}")
    print(f"problems with >=1 pass and >=1 fail (usable for ranking): {n_usable}")
    any_pass = sum(1 for pi in results if any(c["passed"] for c in results[pi]))
    print(f"problems with >=1 pass (best-of-{args.n_samples} oracle): {any_pass}/{len(probs)} = {any_pass/len(probs):.1%}")


if __name__ == "__main__":
    main()
