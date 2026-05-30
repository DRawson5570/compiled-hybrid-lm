#!/usr/bin/env python3
"""Correct HumanEval pass@1 harness for Qwen2.5-Coder-*-Instruct.

Fixes the bugs in train_full.py / debug_eval.py that produced a fake 0/40:
  - Uses the chat template (these are -Instruct models, not completion models).
  - Robust code-block extraction (strips ```python fences / prose).
  - Adequate generation length (512 new tokens), proper EOS handling.
  - Executes generated program against the HumanEval test in a subprocess sandbox.

This measures the TRUE base-model baseline (no cartridge). Run this before
drawing any conclusion about whether a cartridge helps.
"""
import argparse
import os
import re
import subprocess
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"

INSTRUCTION = (
    "Complete the following Python function. "
    "Return ONLY the complete function (including the signature) inside a single "
    "```python code block. Do not include explanations, examples, or test code.\n\n"
    "```python\n{prompt}\n```"
)


def extract_code(text: str) -> str:
    """Pull the python code out of a chat completion."""
    # Drop any thinking block emitted by reasoning models.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Prefer a fenced ```python ... ``` block.
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    # No fence: assume the whole thing is code.
    return text


def build_program(prompt: str, code: str, entry_point: str) -> str:
    """Assemble a runnable program from the extracted code."""
    code = code.strip("\n")
    # If the model re-emitted the signature, use its code as the full program.
    if f"def {entry_point}" in code:
        return code
    # Otherwise treat it as a body completion appended to the original prompt.
    return f"{prompt}{code}"


def run_test(program: str, test_code: str, entry_point: str, timeout: float = 10.0) -> bool:
    full = f"{program}\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full)
        tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=0, help="0 = all 164 problems")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--thinking", action="store_true", help="enable reasoning CoT (slower)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.model} on {args.device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map={"": args.device},
        trust_remote_code=True,
    ).eval()

    he = load_dataset("openai_humaneval", split="test")
    problems = list(he)
    if args.n > 0:
        problems = problems[: args.n]

    passes = 0
    fails = []
    for i, ex in enumerate(problems):
        prompt = ex["prompt"]
        entry = ex["entry_point"]
        messages = [{"role": "user", "content": INSTRUCTION.format(prompt=prompt)}]
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.thinking,
            )
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = extract_code(gen)
        program = build_program(prompt, code, entry)
        ok = run_test(program, ex["test"], entry)
        passes += int(ok)
        if not ok:
            fails.append(ex["task_id"])
        if args.verbose and not ok:
            print(f"\n--- FAIL {ex['task_id']} ---\n{gen[:400]}\n", flush=True)
        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
            print(f"[{i+1}/{len(problems)}] pass={passes} ({passes/(i+1):.1%})", flush=True)

    rate = passes / len(problems)
    print(f"\n=== {args.model} ===")
    print(f"HumanEval pass@1 (greedy): {passes}/{len(problems)} = {rate:.1%}")
    if fails:
        print(f"Failures: {fails}")


if __name__ == "__main__":
    main()
