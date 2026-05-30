#!/usr/bin/env python3
"""Lottery-ticket analysis: which cartridge inject layers help vs hurt?

Zeroes each injection layer's gamma in turn, evaluates on the HumanEval
problems the cartridge FIXED and BROKE. Identifies layers whose removal
recovers breaks without losing fixes — candidates for pruning.

Usage:
    python lottery.py --ckpt path/to/cartridge_best.pt --device cuda:0
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "deepseek_experiments"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack

MODEL = "Qwen/Qwen3.5-4B"
INSTRUCTION = (
    "Complete the following Python function. Return ONLY the complete function "
    "(including the signature) inside a single ```python code block, no explanation.\n\n"
    "```python\n{prompt}\n```"
)
AB_EVAL_PATH = Path.home() / "deepseek_experiments/artifacts/qwen35_4b_rft/ab_eval.json"


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def run_test(code, prob):
    code = code.strip("\n")
    program = code if f"def {prob['entry_point']}" in code else f"{prob['prompt']}{code}"
    full = f"{program}\n\n{prob['test']}\n\ncheck({prob['entry_point']})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full)
        tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=10.0)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


class LotteryAnalyzer:
    def __init__(self, ckpt_path, device="cuda:0"):
        self.device = device
        self.ckpt = torch.load(ckpt_path, map_location="cpu")
        self.inject_layers = self.ckpt["inject_layers"]

        self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16, device_map={"": device}, trust_remote_code=True).eval()

        d = self.model.config.hidden_size
        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=d, inject_layers=self.inject_layers, bottleneck=128,
            init_scale=0.005, noise_scale=0.0, semantic_dim=16).to(device).float()
        self.steerer.load_state_dict(self.ckpt["steerer_state"])
        self.steerer.eval()

        self.manifest = CartridgeManifest(
            "lottery-test", CartridgeRole.DOMAIN_CAPABILITY, MODEL, MODEL,
            steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(self.inject_layers),
            parameter_count=sum(p.numel() for p in self.steerer.parameters()))
        self.rack = SteererCartridgeRack()
        self.rack.mount(self.manifest, self.steerer, weight=1.0, active=True)
        self.rack.register_hooks(self.model.model)

        print(f"Loaded: step={self.ckpt.get('step','?')} "
              f"rate={self.ckpt.get('rate','?')} inject={self.inject_layers}", flush=True)

        # Show current gammas
        print("\nCurrent gammas:")
        for layer, g in sorted(self.steerer.gammas.items()):
            print(f"  layer {str(layer):>3s}: gamma={g.item():+.4f}", flush=True)

    def set_active(self, on):
        self.rack.activate(self.manifest.cartridge_id, on)

    def set_gamma(self, layer, value):
        self.steerer.gammas[str(layer)].data.fill_(value)

    def chat(self, prompt):
        msgs = [{"role": "user", "content": INSTRUCTION.format(prompt=prompt)}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def evaluate(self, problems, max_new=512):
        prompts = [self.chat(p["prompt"]) for p in problems]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        o = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=self.tok.eos_token_id)
        gens = self.tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        results = {}
        for p, g in zip(problems, gens):
            results[p["task_id"]] = run_test(extract_code(g), p)
        return results

    def cleanup(self):
        self.rack.remove_hooks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(Path.home() / "deepseek_experiments/artifacts/qwen35_4b_rft/cartridge_best.pt"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ab-eval", default=str(AB_EVAL_PATH))
    args = ap.parse_args()

    # Load A/B eval to get fixed/broken lists
    ab = json.loads(Path(args.ab_eval).read_text())
    fixed_ids = set(ab["fixed"])
    broke_ids = set(ab["broke"])
    print(f"Fixed: {len(fixed_ids)}  Broken: {len(broke_ids)}", flush=True)

    # Load problems
    all_probs = {e["task_id"]: {"task_id": e["task_id"], "prompt": e["prompt"],
                                 "test": e["test"], "entry_point": e["entry_point"]}
                 for e in load_dataset("openai_humaneval", split="test")}
    target_ids = list(fixed_ids | broke_ids)
    target_probs = [all_probs[tid] for tid in target_ids if tid in all_probs]
    print(f"Target problems: {len(target_probs)}", flush=True)

    la = LotteryAnalyzer(args.ckpt, args.device)

    # Full cartridge baseline
    print("\n=== FULL CARTRIDGE (all gammas active) ===", flush=True)
    for k, v in la.steerer.gammas.items():
        v.data.fill_(float(la.ckpt["steerer_state"][f"gammas.{k}"]))
    la.set_active(True)
    full = la.evaluate(target_probs)
    full_broke = sum(1 for tid in broke_ids if full.get(tid))
    full_fixed = sum(1 for tid in fixed_ids if full.get(tid))
    print(f"  Fixed preserved: {full_fixed}/{len(fixed_ids)}  "
          f"Broken recovered: {full_broke}/{len(broke_ids)}", flush=True)

    # Layer-by-layer zeroing test
    print("\n=== LAYER ZEROING ===", flush=True)
    results = []
    for layer in la.inject_layers:
        # Restore all gammas from checkpoint
        for k, v in la.steerer.gammas.items():
            v.data.fill_(float(la.ckpt["steerer_state"][f"gammas.{k}"]))
        # Zero just this layer
        la.set_gamma(layer, 0.0)

        ev = la.evaluate(target_probs)
        broke_recover = sum(1 for tid in broke_ids if ev.get(tid))
        fixed_keep = sum(1 for tid in fixed_ids if ev.get(tid))
        broke_delta = broke_recover - full_broke
        fixed_delta = fixed_keep - full_fixed

        tag = ""
        if broke_delta > 0 and fixed_delta >= 0:
            tag = "WINNER"  # recovers breaks, keeps fixes
        elif broke_delta > 0:
            tag = "MIXED"
        elif fixed_delta < 0:
            tag = "HURTS"

        results.append((layer, broke_recover, broke_delta, fixed_keep, fixed_delta, tag))
        print(f"  layer {layer:2d} gamma=0: "
              f"broke_rec={broke_recover}/{len(broke_ids)} ({broke_delta:+d})  "
              f"fixed_keep={fixed_keep}/{len(fixed_ids)} ({fixed_delta:+d})  {tag}", flush=True)

    # Summary
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"  Full cartridge: fixed_keep={full_fixed} broke_rec={full_broke}", flush=True)
    print(f"  Inject layers: {la.inject_layers}", flush=True)
    winners = [(l, br, bd, fk, fd) for l, br, bd, fk, fd, t in results if t == "WINNER"]
    if winners:
        print(f"  WINNER layers (zeroing helps): {[l for l,_,_,_,_ in winners]}", flush=True)
    else:
        print(f"  No single layer is a clear winner", flush=True)

    # Also try: zero all HURTS layers together
    hurt_layers = [l for l,_,_,_,_,t in results if t == "HURTS"]
    if hurt_layers:
        print(f"\n  Testing: zero ALL hurt layers {hurt_layers}", flush=True)
        for k, v in la.steerer.gammas.items():
            v.data.fill_(float(la.ckpt["steerer_state"][f"gammas.{k}"]))
        for l in hurt_layers:
            la.set_gamma(l, 0.0)
        ev = la.evaluate(target_probs)
        br = sum(1 for tid in broke_ids if ev.get(tid))
        fk = sum(1 for tid in fixed_ids if ev.get(tid))
        print(f"  Zero {hurt_layers}: fixed_keep={fk}/{len(fixed_ids)}  "
              f"broke_rec={br}/{len(broke_ids)}  "
              f"(delta: fixed={fk-full_fixed:+d} broke={br-full_broke:+d})", flush=True)

    la.cleanup()


if __name__ == "__main__":
    main()
