"""Phase 4: Two-phase per-weakness cartridge training.

Phase A (canonical): Masked cross-entropy on DeepSeek-synthesized canonical solutions.
Phase B (rft): Generate candidate completions from target model, train only on
               the model's own passing rollouts (RFT/STaR).
Evaluates on the weakness's specific HumanEval problem set.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path.home() / "deepseek_experiments"))
sys.path.insert(0, str(Path.home() / "code_harness"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack


def _run_code_test(generated_code: str, test_code: str, timeout: float = 10.0) -> bool:
    full = f"{generated_code}\n\n{test_code}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(full)
        tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0 and "FAIL" not in r.stdout and "FAIL" not in r.stderr
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


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


class WeaknessCartridgeTrainer:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-4B", device: str = "cuda:0",
                 inject_layers: list[int] | None = None, bottleneck: int = 128,
                 seq_cap: int = 128, use_gc: bool = True):
        self.model_name = model_name
        self.device = torch.device(device)
        self.seq_cap = seq_cap
        self.use_gc = use_gc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map={"": device},
            trust_remote_code=True).eval()

        for p in self.model.parameters():
            p.requires_grad = False

        if use_gc:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        self.d_model = self.model.config.hidden_size
        n_layers = self.model.config.num_hidden_layers
        if inject_layers is None:
            inject_layers = [i for i in (0, 2, 4, 7, 10, 14, 17, 21, 24, 26) if i < n_layers]
        self.inject_layers = inject_layers

        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=self.d_model, inject_layers=self.inject_layers,
            bottleneck=bottleneck, init_scale=0.005, noise_scale=0.0).to(self.device)
        for g in self.steerer.gammas.values():
            g.data.fill_(0.02)

        self.manifest = CartridgeManifest(
            "weakness-cartridge", CartridgeRole.DOMAIN_CAPABILITY,
            model_name, model_name, steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(self.inject_layers),
            parameter_count=sum(p.numel() for p in self.steerer.parameters()),
        )
        self.rack = SteererCartridgeRack()
        self.rack.mount(self.manifest, self.steerer, weight=1.0, active=True)
        self.rack.register_hooks(self.model.model)
        self.cartridge_id = self.manifest.cartridge_id
        print(f"d={self.d_model} inject={self.inject_layers} "
              f"params={sum(p.numel() for p in self.steerer.parameters()):,}", flush=True)

    def _set_weights(self, seq_len: int):
        dev = next(self.model.parameters()).device
        self.rack.set_weights(torch.zeros(1, seq_len, 21, device=dev))

    def set_active(self, on: bool):
        self.rack.activate(self.cartridge_id, on)

    def _format_chat(self, content: str) -> str:
        msgs = [{"role": "user", "content": content}]
        try:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _train_on_example(self, prompt_raw: str, target_code: str) -> float:
        self.steerer.train()
        content = HE_INSTRUCTION.format(prompt=prompt_raw)
        chat_prompt = self._format_chat(content)
        prompt_ids = self.tokenizer.encode(chat_prompt)
        target_ids = self.tokenizer.encode(target_code)
        full_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
        full_ids = full_ids[:self.seq_cap]
        if len(full_ids) < 2:
            return 0.0

        dev = next(self.model.parameters()).device
        x = torch.tensor([full_ids[:-1]], device=dev)
        y = torch.tensor([full_ids[1:]], device=dev)

        mask = torch.zeros_like(y, dtype=torch.float32)
        prompt_ctx = max(0, len(prompt_ids) - 1)
        mask[:, prompt_ctx:] = 1.0

        self._set_weights(x.shape[1])
        if self.use_gc:
            self.model.train()
        try:
            logits = self.model(input_ids=x).logits.float()
        finally:
            if self.use_gc:
                self.model.eval()

        loss_tokens = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            reduction="none").reshape_as(mask)
        loss = (loss_tokens * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss + 0.00005 * self.steerer.orthogonal_penalty()
        return loss

    def train_step_canonical(self, prompt: str, expected: str) -> float:
        return self._train_on_example(prompt, expected)

    @torch.no_grad()
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        ids = list(self.tokenizer.encode(prompt))
        self.steerer.eval()
        self.model.eval()
        generated_ids: list[int] = []
        for i in range(max_tokens):
            dev = next(self.model.parameters()).device
            ctx = ids + generated_ids
            x = torch.tensor([ctx[-min(len(ctx), 512):]], device=dev)
            self._set_weights(x.shape[1])
            out = self.model(x)
            logits = out.logits[0, -1].float().cpu()
            if not torch.isfinite(logits).all():
                break
            nid = int(logits.argmax())
            if nid in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
                break
            generated_ids.append(nid)
            if i > 3 and i % 4 == 0:
                current = self.tokenizer.decode(generated_ids)
                if current.endswith("\n\n") and not current.rstrip().endswith(":") \
                   and (current.count("\n\n") >= 3 or i > 30):
                    break
        return self.tokenizer.decode(generated_ids)

    def evaluate_on_problems(self, problems: list[dict[str, str]]) -> dict[str, Any]:
        self.steerer.eval()
        self.set_active(True)
        passes = 0
        results: list[dict[str, Any]] = []
        for p in problems:
            content = HE_INSTRUCTION.format(prompt=p["prompt"])
            chat_prompt = self._format_chat(content)
            gen = self.generate(chat_prompt, max_tokens=256)
            code = extract_code(gen)
            passed = run_humaneval_test(p["prompt"], code, p["test"], p["entry_point"])
            if passed:
                passes += 1
            results.append({
                "task_id": p["task_id"],
                "passed": passed,
                "generated": code,
            })
        return {"passes": passes, "total": len(problems),
                "rate": passes / max(len(problems), 1), "results": results}

    def cleanup(self):
        self.rack.remove_hooks()


def train_canonical_phase(trainer: WeaknessCartridgeTrainer, training_data: list[dict],
                          steps: int = 500, lr: float = 3e-4, weight_decay: float = 0.01,
                          eval_problems: list[dict] | None = None,
                          save_dir: Path | None = None) -> dict[str, Any]:
    opt = torch.optim.AdamW(trainer.steerer.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    best_rate = 0.0
    best_state = None

    train_examples = [
        {"prompt": ex["prompt"], "expected": ex["expected"]}
        for ex in training_data if ex.get("prompt") and ex.get("expected")
    ]
    if not train_examples:
        print("  No canonical training examples found, skipping", flush=True)
        return {"best_rate": 0.0, "steps": 0}

    print(f"  Canonical CE on {len(train_examples)} examples, {steps} steps...", flush=True)
    for step in range(1, steps + 1):
        ch = random.choice(train_examples)
        opt.zero_grad()
        loss = trainer.train_step_canonical(ch["prompt"], ch["expected"])
        if loss > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.steerer.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        if step % 100 == 0 and eval_problems:
            avg_loss = sum(losses[-50:]) / min(50, len(losses)) if losses else 0
            trainer.steerer.eval()
            result = trainer.evaluate_on_problems(eval_problems)
            trainer.steerer.train()
            print(f"    [canonical {step:3d}] loss={avg_loss:.4f}  "
                  f"eval={result['passes']}/{result['total']} ({result['rate']:.1%})", flush=True)
            if result["rate"] >= best_rate:
                best_rate = result["rate"]
                best_state = {k: v.detach().cpu().clone()
                             for k, v in trainer.steerer.state_dict().items()}
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"steerer_state": best_state, "step": step, "rate": best_rate,
                                "phase": "canonical"}, save_dir / "canonical_best.pt")

    if best_state is not None:
        trainer.steerer.load_state_dict(best_state)

    if save_dir:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in trainer.steerer.state_dict().items()}
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"steerer_state": final_state, "step": steps, "rate": best_rate,
                    "phase": "canonical"}, save_dir / "canonical_best.pt")

    eval_result = None
    if eval_problems:
        trainer.steerer.eval()
        eval_result = trainer.evaluate_on_problems(eval_problems)
        print(f"  Canonical phase done: eval={eval_result['passes']}/{eval_result['total']} "
              f"({eval_result['rate']:.1%})", flush=True)

    return {"best_rate": best_rate, "steps": steps, "final_eval": eval_result}


def train_rft_phase(trainer: WeaknessCartridgeTrainer, training_data: list[dict],
                    steps: int = 500, lr: float = 3e-4, weight_decay: float = 0.01,
                    eval_problems: list[dict] | None = None,
                    save_dir: Path | None = None) -> dict[str, Any]:
    opt = torch.optim.AdamW(trainer.steerer.parameters(), lr=lr, weight_decay=weight_decay)
    best_rate = 0.0
    best_state = None

    train_examples = [
        {"prompt": ex["prompt"], "expected": ex["expected"]}
        for ex in training_data if ex.get("prompt") and ex.get("expected")
    ]
    if not train_examples:
        print("  No RFT training examples found, skipping", flush=True)
        return {"best_rate": 0.0, "steps": 0}

    print(f"  RFT on {len(train_examples)} problems, {steps} steps "
          f"(lazy 1-candidate-per-step)...", flush=True)

    prompt_to_test = {ex["prompt"]: ex.get("test_code", "")
                      for ex in training_data if ex.get("prompt")}

    for step in range(1, steps + 1):
        ch = random.choice(train_examples)
        content = HE_INSTRUCTION.format(prompt=ch["prompt"])
        chat_prompt = trainer._format_chat(content)

        trainer.set_active(True)
        trainer.steerer.train()
        gen = trainer.generate(chat_prompt, max_tokens=256)
        code = extract_code(gen)

        test_code = prompt_to_test.get(ch["prompt"], "")
        if test_code:
            passed = _run_code_test(code, test_code)
        else:
            passed = False

        opt.zero_grad()
        if passed:
            loss = trainer._train_on_example(ch["prompt"], code)
        else:
            loss = trainer._train_on_example(ch["prompt"], ch["expected"])
        if loss > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.steerer.parameters(), 1.0)
            opt.step()

        if step % 100 == 0 and eval_problems:
            trainer.steerer.eval()
            eval_result = trainer.evaluate_on_problems(eval_problems)
            trainer.steerer.train()
            print(f"    [rft {step:3d}] eval={eval_result['passes']}/{eval_result['total']} "
                  f"({eval_result['rate']:.1%})", flush=True)
            if eval_result["rate"] >= best_rate:
                best_rate = eval_result["rate"]
                best_state = {k: v.detach().cpu().clone()
                             for k, v in trainer.steerer.state_dict().items()}
                if save_dir:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"steerer_state": best_state, "step": step, "rate": best_rate,
                                "phase": "rft"}, save_dir / "rft_best.pt")

    if best_state is not None:
        trainer.steerer.load_state_dict(best_state)

    if save_dir:
        final_state = {k: v.detach().cpu().clone()
                       for k, v in trainer.steerer.state_dict().items()}
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"steerer_state": final_state, "step": steps, "rate": best_rate,
                    "phase": "rft"}, save_dir / "rft_best.pt")

    eval_result = None
    if eval_problems:
        trainer.steerer.eval()
        eval_result = trainer.evaluate_on_problems(eval_problems)
        print(f"  RFT phase done: eval={eval_result['passes']}/{eval_result['total']} "
              f"({eval_result['rate']:.1%})", flush=True)

    return {"best_rate": best_rate, "steps": steps, "final_eval": eval_result}


def load_humaneval_problems(task_ids: list[str] | None = None) -> list[dict[str, str]]:
    from datasets import load_dataset
    problems = [
        {"task_id": e["task_id"], "prompt": e["prompt"], "test": e["test"],
         "entry_point": e["entry_point"], "canonical_solution": e["canonical_solution"]}
        for e in load_dataset("openai/openai_humaneval", split="test")
    ]
    if task_ids:
        problems = [p for p in problems if p["task_id"] in set(task_ids)]
    return problems


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weakness-id", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--catalog", default=str(Path.home() / "code_harness" / "weaknesses" / "catalog.json"))
    ap.add_argument("--training-dir", default=str(Path.home() / "code_harness" / "weaknesses"))
    ap.add_argument("--output-dir", default=str(Path.home() / "code_harness" / "artifacts" / "cartridges"))
    ap.add_argument("--mode", choices=["canonical", "rft", "both"], default="both")
    ap.add_argument("--canonical-steps", type=int, default=500)
    ap.add_argument("--rft-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-gc", action="store_true", help="Disable gradient checkpointing")
    args = ap.parse_args()

    catalog = json.loads(Path(args.catalog).read_text())
    w = next((w for w in catalog.get("weaknesses", []) if w["weakness_id"] == args.weakness_id), None)
    if w is None:
        print(f"Weakness not found: {args.weakness_id}")
        return

    training_path = Path(args.training_dir) / f"{args.weakness_id}_training.jsonl"
    training_data = []
    if training_path.exists():
        with open(training_path) as f:
            training_data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(training_data)} training examples from {training_path}")
    if not training_data:
        print(f"  WARNING: No training examples found for {args.weakness_id}.")
        print(f"  Expected file: {training_path}")
        print(f"  Run 'python teacher/synthesize.py --weakness-id {args.weakness_id}' first.")
        sys.exit(1)

    eval_task_ids = w.get("failing_task_ids", [])
    eval_problems = load_humaneval_problems(eval_task_ids)
    print(f"Eval problems for this weakness: {len(eval_problems)}")

    save_dir = Path(args.output_dir) / args.weakness_id

    trainer = WeaknessCartridgeTrainer(model_name=args.model, device=args.device,
                                        use_gc=not args.no_gc)

    print(f"\n=== Baseline (no cartridge) ===", flush=True)
    trainer.set_active(False)
    trainer.steerer.eval()
    bl_result = trainer.evaluate_on_problems(eval_problems)
    print(f"  Baseline: {bl_result['passes']}/{bl_result['total']} ({bl_result['rate']:.1%})", flush=True)

    all_results = {"weakness_id": args.weakness_id, "baseline": bl_result}

    if args.mode in ("canonical", "both"):
        print(f"\n=== Phase A: Canonical CE ===", flush=True)
        canonical_result = train_canonical_phase(
            trainer, training_data, steps=args.canonical_steps, lr=args.lr,
            eval_problems=eval_problems, save_dir=save_dir)
        all_results["canonical"] = canonical_result

    if args.mode in ("rft", "both"):
        print(f"\n=== Phase B: RFT Refinement ===", flush=True)
        rft_result = train_rft_phase(
            trainer, training_data, steps=args.rft_steps, lr=args.lr,
            eval_problems=eval_problems, save_dir=save_dir)
        all_results["rft"] = rft_result

    final_result = trainer.evaluate_on_problems(eval_problems)
    print(f"\n=== Final ===", flush=True)
    print(f"  Baseline: {bl_result['passes']}/{bl_result['total']} ({bl_result['rate']:.1%})", flush=True)
    print(f"  Final:    {final_result['passes']}/{final_result['total']} ({final_result['rate']:.1%})", flush=True)
    delta = final_result['passes'] - bl_result['passes']
    print(f"  Delta:    {delta:+d}", flush=True)
    all_results["final"] = final_result
    all_results["delta"] = delta

    best_state = {k: v.detach().cpu().clone() for k, v in trainer.steerer.state_dict().items()}
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "steerer_state": best_state,
        "inject_layers": trainer.inject_layers,
        "semantic_dim": trainer.steerer.semantic_dim,
        "bottleneck": trainer.steerer.bottleneck,
        "extra_channels": trainer.steerer.extra_channels,
        "weakness_id": args.weakness_id,
        "results": all_results,
    }, save_dir / "cartridge_best.pt")
    (save_dir / "eval.json").write_text(json.dumps(all_results, indent=2))
    print(f"Saved to {save_dir}", flush=True)

    trainer.cleanup()


if __name__ == "__main__":
    main()
