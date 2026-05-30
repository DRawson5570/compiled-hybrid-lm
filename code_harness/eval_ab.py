#!/usr/bin/env python3
"""A/B eval: base model vs trained adapter cartridge on full HumanEval.

Loads a cartridge_best.pt produced by train_rft.py, mounts it on the frozen
base, and measures greedy pass@1 with the cartridge OFF (true base) vs ON.
Reports the delta and the exact set of problems the cartridge FIXES (base fail
-> cartridge pass) and BREAKS (base pass -> cartridge fail). This is the
definitive answer to "does the training approach improve pass@1".
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.home() / "deepseek_experiments"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack

MODEL = "Qwen/Qwen3.5-4B"
INSTRUCTION = (
    "Complete the following Python function. Return ONLY the complete function "
    "(including the signature) inside a single ```python code block, no explanation.\n\n"
    "```python\n{prompt}\n```"
)


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def build_program(prompt, code, entry):
    code = code.strip("\n")
    return code if f"def {entry}" in code else f"{prompt}{code}"


def run_test(program, test_code, entry, timeout=10.0):
    full = f"{program}\n\n{test_code}\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full)
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


class ABEval:
    def __init__(self, ckpt_path, device="cuda:0"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # Must match the 4-bit base the cartridge was trained on for a fair A/B.
        qconf = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=qconf, device_map={"": device},
            dtype=torch.float16, trust_remote_code=True).eval()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.inject_layers = ckpt["inject_layers"]
        d = self.model.config.hidden_size
        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=d, inject_layers=self.inject_layers, bottleneck=128,
            init_scale=0.005, noise_scale=0.0, semantic_dim=16).to(device)
        self.steerer.float()
        self.steerer.load_state_dict(ckpt["steerer_state"])
        self.steerer.eval()
        self.manifest = CartridgeManifest(
            "code-rft-v1", CartridgeRole.DOMAIN_CAPABILITY, MODEL, MODEL,
            steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(self.inject_layers),
            parameter_count=sum(p.numel() for p in self.steerer.parameters()))
        self.rack = SteererCartridgeRack()
        self.rack.mount(self.manifest, self.steerer, weight=1.0, active=True)
        self.rack.register_hooks(self.model.model)
        print(f"loaded cartridge step={ckpt.get('step')} train_rate={ckpt.get('rate')}", flush=True)

    def chat(self, content):
        msgs = [{"role": "user", "content": content}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def gen(self, prompts, max_new=512, batch=16):
        torch.cuda.empty_cache()
        outs = []
        for b in range(0, len(prompts), batch):
            chunk = prompts[b:b + batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            o = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
            outs.extend(self.tok.batch_decode(o[:, enc["input_ids"].shape[1]:],
                                              skip_special_tokens=True))
        torch.cuda.empty_cache()
        return outs

    def run(self, probs, active, batch=16):
        self.rack.activate(self.manifest.cartridge_id, active)
        prompts = [self.chat(INSTRUCTION.format(prompt=p["prompt"])) for p in probs]
        gens = self.gen(prompts, batch=batch)
        res = {}
        for p, g in zip(probs, gens):
            res[p["task_id"]] = run_test(build_program(p["prompt"], extract_code(g), p["entry_point"]),
                                         p["test"], p["entry_point"])
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path.home() / "deepseek_experiments/artifacts/qwen35_4b_rft/cartridge_best.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    probs = [{"task_id": e["task_id"], "prompt": e["prompt"], "test": e["test"],
              "entry_point": e["entry_point"]} for e in load_dataset("openai/openai_humaneval", split="test")]
    if args.n > 0:
        probs = probs[: args.n]

    ab = ABEval(args.ckpt, args.device)
    base = ab.run(probs, active=False, batch=args.batch)
    cart = ab.run(probs, active=True, batch=args.batch)

    n = len(probs)
    b = sum(base.values())
    c = sum(cart.values())
    fixed = sorted(t for t in base if cart[t] and not base[t])
    broke = sorted(t for t in base if base[t] and not cart[t])
    print(f"\n=== A/B HumanEval pass@1 (greedy, n={n}) ===")
    print(f"BASE      : {b}/{n} = {b/n:.1%}")
    print(f"CARTRIDGE : {c}/{n} = {c/n:.1%}")
    print(f"DELTA     : {(c-b)/n:+.1%}  ({c-b:+d} problems)")
    print(f"FIXED  (base fail -> cart pass): {fixed}")
    print(f"BROKE  (base pass -> cart fail): {broke}")
    out = Path(args.ckpt).parent / "ab_eval.json"
    out.write_text(json.dumps({"base": b, "cart": c, "n": n, "fixed": fixed, "broke": broke}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
