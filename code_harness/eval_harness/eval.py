#!/usr/bin/env python3
"""Clean eval harness. A/B-compares frozen base vs base+cartridge on code benchmarks.

Usage:
    python eval_harness/eval.py --benchmark humaneval
    python eval_harness/eval.py --benchmark mbpp --split test --n 50
    python eval_harness/eval.py --benchmark humaneval --cartridge path/to/cartridge_best.pt
    python eval_harness/eval.py --benchmark humaneval --model Qwen/Qwen2.5-Coder-7B

Output: eval_harness/results/{benchmark}_{timestamp}.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path.home() / "deepseek_experiments"))
from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack

RESULTS_DIR = Path.home() / "code_harness" / "eval_harness" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompt templates ─────────────────────────────────────────────────────

CHAT_INSTRUCTION = (
    "Complete the following Python function. Return ONLY the complete function "
    "(including the signature) inside a single ```python code block, no explanation.\n\n"
    "```python\n{prompt}\n```"
)

RAW_INSTRUCTION = "{prompt}"


# ── Code extraction ──────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


# ── Test execution ───────────────────────────────────────────────────────

def run_test(program: str, test_code: str, entry_point: str,
             timeout: float = 10.0) -> bool:
    """Run HumanEval-style test. MBPP uses entry_point=None and embeds tests in program."""
    full = f"{program}\n\n{test_code}\n\ncheck({entry_point})\n" if entry_point else program
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(full)
        tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], capture_output=True, text=True,
                          timeout=timeout)
        return r.returncode == 0 and "FAIL" not in r.stdout and "FAIL" not in r.stderr
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── Benchmark loaders ────────────────────────────────────────────────────

def load_humaneval(split="test") -> list[dict]:
    """Returns [{task_id, prompt, test, entry_point}]."""
    return [{"task_id": e["task_id"], "prompt": e["prompt"],
             "test": e["test"], "entry_point": e["entry_point"]}
            for e in load_dataset("openai_humaneval", split=split)]


def load_mbpp(split="test") -> list[dict]:
    """Returns [{task_id, prompt, test_list, test_imports, entry_point=None}]."""
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    return [{"task_id": f"mbpp/{e['task_id']}", "prompt": e["prompt"].strip(),
             "code": e["code"].strip(), "test_list": e["test_list"],
             "test_imports": e.get("test_imports", []) or []}
            for e in ds]


# ── Program builders ─────────────────────────────────────────────────────

def build_he_program(prompt: str, code: str, entry_point: str) -> str:
    code = code.strip("\n")
    if not code:
        return ""
    return (code if f"def {entry_point}" in code else f"{prompt}{code}")


def build_mbpp_program(code: str, prob: dict) -> str:
    code = code.strip("\n")
    if not code:
        return ""
    setup = "\n".join(prob["test_imports"])
    tests = "\n".join(prob["test_list"])
    return f"{setup}\n{code}\n{tests}\n"


# ── Generator ────────────────────────────────────────────────────────────

class CodeEvaluator:
    """Load model, optionally mount cartridge, run A/B benchmark."""

    def __init__(self, model_name: str, device: str = "cuda:0",
                 use_chat_template: bool = True,
                 cartridge_path: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.use_chat = use_chat_template
        self.cartridge_loaded = cartridge_path is not None

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map={"": device},
            trust_remote_code=True).eval()

        self.d_model = self.model.config.hidden_size
        self.rack = SteererCartridgeRack()
        self.steerer = None
        self.cartridge_id = "eval-cartridge"

        if cartridge_path:
            self._mount_cartridge(cartridge_path)

    def _mount_cartridge(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        inject_layers = ckpt["inject_layers"]
        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=self.d_model, inject_layers=inject_layers,
            bottleneck=128, init_scale=0.005, noise_scale=0.0,
            semantic_dim=16).to(self.device).float()
        self.steerer.load_state_dict(ckpt["steerer_state"])
        self.steerer.eval()
        print(f"  cartridge: step={ckpt.get('step','?')} "
              f"train_rate={ckpt.get('rate','?')} inject={inject_layers}",
              flush=True)
        manifest = CartridgeManifest(
            self.cartridge_id, CartridgeRole.DOMAIN_CAPABILITY,
            self.model_name, self.model_name,
            steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(inject_layers),
            parameter_count=sum(p.numel() for p in self.steerer.parameters()))
        self.rack.mount(manifest, self.steerer, weight=1.0, active=False)
        self.rack.register_hooks(self.model.model)

    def set_active(self, on: bool):
        if self.cartridge_loaded:
            self.rack.activate(self.cartridge_id, on)

    def _format_prompt(self, prob: dict, instruction_template: str) -> str:
        content = instruction_template.format(prompt=prob["prompt"])
        if self.use_chat:
            msgs = [{"role": "user", "content": content}]
            try:
                return self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                return self.tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True)
        return content

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new: int = 512,
                 batch: int = 8) -> list[str]:
        """Batched greedy generation."""
        outs = []
        for b in range(0, len(prompts), batch):
            chunk = prompts[b:b + batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            o = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
            gen = o[:, enc["input_ids"].shape[1]:]
            outs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return outs

    def evaluate(self, problems: list[dict], active: bool, max_new: int = 512,
                 batch: int = 8) -> dict[str, bool]:
        """Return {task_id: pass_bool}."""
        self.set_active(active)
        prompts = [self._format_prompt(p, CHAT_INSTRUCTION) for p in problems]
        gens = self.generate(prompts, max_new=max_new, batch=batch)

        results = {}
        for p, g in zip(problems, gens):
            code = extract_code(g)
            if p.get("entry_point"):
                program = build_he_program(p["prompt"], code, p["entry_point"])
                passed = run_test(program, p["test"], p["entry_point"])
            else:
                program = build_mbpp_program(code, p)
                passed = run_test(program, "", None)
            results[p["task_id"]] = passed
        return results

    def cleanup(self):
        if self.cartridge_loaded:
            self.rack.remove_hooks()


# ── Report ───────────────────────────────────────────────────────────────

def build_report(problems: list[dict], base: dict[str, bool],
                 cart: dict[str, bool] | None,
                 model_name: str, benchmark: str,
                 cartridge_path: str | None) -> dict:
    n = len(problems)
    base_passes = sum(base.values())
    base_rate = base_passes / n

    report = {
        "model": model_name,
        "benchmark": benchmark,
        "n": n,
        "cartridge": cartridge_path,
        "base": {"passes": base_passes, "total": n, "rate": base_rate},
    }

    if cart is not None:
        cart_passes = sum(cart.values())
        cart_rate = cart_passes / n
        delta = cart_passes - base_passes
        fixed = sorted(k for k in base if cart[k] and not base[k])
        broke = sorted(k for k in base if base[k] and not cart[k])
        report["cartridge"] = {"passes": cart_passes, "total": n, "rate": cart_rate}
        report["delta"] = {"absolute": delta, "relative_pct": delta / n * 100}
        report["fixed"] = fixed
        report["broken"] = broke
        report["summary"] = (
            f"base={base_rate:.1%} cart={cart_rate:.1%} "
            f"delta={delta:+d} ({delta/n:+.1%}) "
            f"fixed={len(fixed)} broke={len(broke)}"
        )
    else:
        report["summary"] = f"base={base_rate:.1%}"

    return report


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Code eval harness — A/B base vs cartridge")
    ap.add_argument("--benchmark", choices=["humaneval", "mbpp"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--cartridge", default=None, help="Path to cartridge_best.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=0, help="Limit problems (0=all)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--split", default="test", help="MBPP split: train/test/validation")
    ap.add_argument("--no-chat", action="store_true", help="Disable chat template")
    ap.add_argument("--base-only", action="store_true", help="Base only, no cartridge eval")
    args = ap.parse_args()

    print(f"Loading benchmark: {args.benchmark}", flush=True)
    if args.benchmark == "humaneval":
        problems = load_humaneval()
    else:
        problems = load_mbpp(args.split)

    if args.n > 0:
        problems = problems[:args.n]
    print(f"  {len(problems)} problems", flush=True)

    print(f"Loading model: {args.model}", flush=True)
    ev = CodeEvaluator(
        args.model, args.device,
        use_chat_template=not args.no_chat,
        cartridge_path=args.cartridge)

    t0 = time.time()

    print("\n=== BASE (no cartridge) ===", flush=True)
    base_results = ev.evaluate(problems, active=False, max_new=args.max_new,
                               batch=args.batch)
    base_passes = sum(base_results.values())
    print(f"  pass@1: {base_passes}/{len(problems)} = {base_passes/len(problems):.1%}",
          flush=True)

    cart_results = None
    if not args.base_only and ev.cartridge_loaded:
        print("\n=== CARTRIDGE ===", flush=True)
        cart_results = ev.evaluate(problems, active=True, max_new=args.max_new,
                                   batch=args.batch)
        cart_passes = sum(cart_results.values())
        delta = cart_passes - base_passes
        print(f"  pass@1: {cart_passes}/{len(problems)} = {cart_passes/len(problems):.1%}",
              flush=True)
        print(f"  delta: {delta:+d} ({delta/len(problems):+.1%})", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s", flush=True)

    report = build_report(problems, base_results, cart_results,
                          args.model, args.benchmark, args.cartridge)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{args.benchmark}_{ts}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out}", flush=True)
    print(report["summary"])

    ev.cleanup()


if __name__ == "__main__":
    main()
