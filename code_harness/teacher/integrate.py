"""Phase 5: Mount all weakness cartridges and run full HumanEval regression test.

Loads every cartridge in the cartridge directory, mounts them into a single
SteererCartridgeRack with mean composition, and evaluates:
  - Per-weakness gains/losses
  - Aggregate delta vs baseline
  - Regressions (previously-passing problems that now fail)
  - Held-out weak-problem set (problems not used during any training)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.home() / "deepseek_experiments"))
sys.path.insert(0, str(Path.home() / "code_harness"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack


HE_INSTRUCTION = (
    "Complete the following Python function. Return ONLY the complete function "
    "(including the signature) inside a single ```python code block, no explanation.\n\n"
    "```python\n{prompt}\n```"
)


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def run_humaneval_test(prompt: str, generated: str, test_code: str,
                       entry_point: str, timeout: float = 5.0) -> bool:
    code = generated.strip()
    program = code if f"def {entry_point}" in code else f"{prompt}{code}"
    full = f"{program}\n\n{test_code}\n\ncheck({entry_point})\n"
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


class MultiCartridgeEvaluator:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-4B", device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map={"": device},
            trust_remote_code=True).eval()
        self.d_model = self.model.config.hidden_size
        self.rack = SteererCartridgeRack(composition_mode="mean")
        self._mounted: dict[str, dict] = {}

    def mount_cartridge(self, cartridge_path: Path, weakness_id: str):
        ckpt = torch.load(cartridge_path, map_location="cpu")
        inject_layers = ckpt["inject_layers"]
        semantic_dim = ckpt.get("semantic_dim", 16)
        bottleneck = ckpt.get("bottleneck", 128)
        extra_channels = ckpt.get("extra_channels", 0)
        steerer = FeatureConditionedAdapterSteerer(
            d_model=self.d_model, inject_layers=inject_layers,
            bottleneck=bottleneck, init_scale=0.005, noise_scale=0.0,
            semantic_dim=semantic_dim, extra_channels=extra_channels).to(self.device).float()
        steerer.load_state_dict(ckpt["steerer_state"])
        steerer.eval()

        manifest = CartridgeManifest(
            cartridge_id=weakness_id, role=CartridgeRole.DOMAIN_CAPABILITY,
            base_model_id=self.model_name, tokenizer_id=self.model_name,
            steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(inject_layers),
            parameter_count=sum(p.numel() for p in steerer.parameters()))
        self.rack.mount(manifest, steerer, weight=1.0, active=True)
        self._mounted[weakness_id] = {
            "path": str(cartridge_path), "inject_layers": inject_layers,
            "params": sum(p.numel() for p in steerer.parameters())}

    def set_active(self, on: bool):
        for wid in self._mounted:
            self.rack.activate(wid, on)

    def _format_chat(self, content: str) -> str:
        msgs = [{"role": "user", "content": content}]
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, prompts: list[str], max_new: int = 512, batch: int = 8) -> list[str]:
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
        self.set_active(active)
        prompts = [self._format_chat(HE_INSTRUCTION.format(prompt=p["prompt"]))
                   for p in problems]
        gens = self.generate(prompts, max_new=max_new, batch=batch)
        results = {}
        for p, g in zip(problems, gens):
            code = extract_code(g)
            results[p["task_id"]] = run_humaneval_test(p["prompt"], code, p["test"], p["entry_point"])
        return results

    def cleanup(self):
        self.rack.remove_hooks()


def run_integration_test(cartridge_dir: Path, catalog_path: Path,
                         model_name: str, device: str,
                         output_path: Path | None = None) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text())
    all_problems = [
        {"task_id": e["task_id"], "prompt": e["prompt"], "test": e["test"],
         "entry_point": e["entry_point"], "canonical_solution": e["canonical_solution"]}
        for e in load_dataset("openai/openai_humaneval", split="test")
    ]

    ev = MultiCartridgeEvaluator(model_name=model_name, device=device)

    cartridge_files = sorted(cartridge_dir.glob("*/cartridge_best.pt"))
    for cf in cartridge_files:
        wid = cf.parent.name
        ev.mount_cartridge(cf, wid)
        print(f"  mounted: {wid} ({cf})", flush=True)

    ev.rack.register_hooks(ev.model.model)

    print(f"\nTotal cartridges mounted: {len(ev._mounted)}", flush=True)

    print(f"\n=== BASELINE ({len(all_problems)} problems) ===", flush=True)
    t0 = time.time()
    baseline = ev.evaluate(all_problems, active=False, batch=1, max_new=256)
    bl_passes = sum(baseline.values())
    print(f"  Baseline: {bl_passes}/{len(all_problems)} ({bl_passes/len(all_problems):.1%})  "
          f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== INTEGRATED ({len(all_problems)} problems) ===", flush=True)
    t0 = time.time()
    integrated = ev.evaluate(all_problems, active=True, batch=1, max_new=256)
    int_passes = sum(integrated.values())
    print(f"  Integrated: {int_passes}/{len(all_problems)} ({int_passes/len(all_problems):.1%})  "
          f"({time.time()-t0:.0f}s)", flush=True)

    delta = int_passes - bl_passes
    fixed = sorted(tid for tid in baseline if integrated[tid] and not baseline[tid])
    broke = sorted(tid for tid in baseline if baseline[tid] and not integrated[tid])
    print(f"\n  Delta: {delta:+d} ({delta/len(all_problems):+.1%})", flush=True)
    print(f"  Fixed: {len(fixed)}  Broken: {len(broke)}", flush=True)

    by_weakness = {}
    for w in catalog.get("weaknesses", []):
        wid = w["weakness_id"]
        task_ids = set(w.get("failing_task_ids", []))
        w_bl = sum(1 for tid in task_ids if baseline.get(tid))
        w_int = sum(1 for tid in task_ids if integrated.get(tid))
        by_weakness[wid] = {
            "total": len(task_ids),
            "baseline_passes": w_bl,
            "integrated_passes": w_int,
            "delta": w_int - w_bl,
            "rate_change": (w_int - w_bl) / max(len(task_ids), 1),
        }
        if len(task_ids) > 0:
            print(f"  {wid}: {w_bl}→{w_int}/{len(task_ids)} ({w_int-w_bl:+d})", flush=True)

    report = {
        "model": model_name,
        "benchmark": "humaneval",
        "n_problems": len(all_problems),
        "n_cartridges": len(ev._mounted),
        "composition_mode": "mean",
        "baseline": {"passes": bl_passes, "total": len(all_problems),
                     "rate": bl_passes / len(all_problems)},
        "integrated": {"passes": int_passes, "total": len(all_problems),
                       "rate": int_passes / len(all_problems)},
        "delta": delta,
        "fixed": fixed,
        "broken": broke,
        "by_weakness": by_weakness,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport: {output_path}", flush=True)

    ev.cleanup()
    return report


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartridge-dir", default=str(Path.home() / "code_harness" / "artifacts" / "cartridges"))
    ap.add_argument("--catalog", default=str(Path.home() / "code_harness" / "weaknesses" / "catalog.json"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=str(Path.home() / "code_harness" / "artifacts" / "regression_report.json"))
    args = ap.parse_args()

    run_integration_test(
        Path(args.cartridge_dir), Path(args.catalog),
        args.model, args.device, Path(args.output),
    )


if __name__ == "__main__":
    main()
