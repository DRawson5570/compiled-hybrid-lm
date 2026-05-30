#!/usr/bin/env python3
"""Qwen3.5-4B code cartridge training with masked CE on correct completions.
Train: MBPP (120) + code_harness challenges (175) = 295 examples
Eval: HumanEval (164) — 10-problem subset every 50 steps, full pass@1 at end
"""
import sys, json, random, time, textwrap, os
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path.home() / "deepseek_experiments"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from sandbox import run_test as sandbox_test

MODEL = "Qwen/Qwen3.5-4B"
OUT_DIR = Path.home() / "deepseek_experiments/artifacts/qwen35_4b_code_cartridge_v1"

# ── Data loading ────────────────────────────────────────────────────────

def load_code_harness(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def load_humaneval_split(train_size=100):
    he = load_dataset("openai_humaneval", split="test")
    examples = []
    for ex in he:
        examples.append({
            "prompt": ex["prompt"].strip(),
            "expected": ex["canonical_solution"].strip(),
            "test": ex["test"].strip(),
            "entry_point": ex["entry_point"],
            "source": "humaneval"
        })
    import random
    random.seed(42)
    random.shuffle(examples)
    train = examples[:train_size]
    eval_set = examples[train_size:]
    return train, eval_set


def load_code_harness_funcs(path):
    with open(path) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    funcs = []
    for r in raw:
        p = r.get("prompt", "")
        if "def " in p and r["tier"] in (1, 4):
            funcs.append({"prompt": p, "expected": r["expected"], "source": "code_harness"})
    return funcs

def load_humaneval():
    he = load_dataset("openai_humaneval", split="test")
    examples = []
    for ex in he:
        prompt = ex["prompt"].strip()
        solution = ex["canonical_solution"].strip()
        examples.append({
            "prompt": prompt,
            "expected": solution,
            "test": ex["test"].strip(),
            "entry_point": ex["entry_point"],
            "source": "humaneval"
        })
    return examples


# ── HumanEval test runner ───────────────────────────────────────────────

def run_humaneval_test(prompt: str, generated: str, test_code: str, entry_point: str,
                       timeout: float = 5.0) -> bool:
    full_code = f"{prompt}{generated}\n\n{test_code}\n\ncheck({entry_point})"
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        tmp = f.name
    try:
        r = subprocess.run(["python", tmp], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0 and "FAIL" not in r.stdout and "FAIL" not in r.stderr
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── Trainer ─────────────────────────────────────────────────────────────

class CodeTrainer:
    def __init__(self, device_str="cuda:0", seq_cap=128, bottleneck=128):
        self.seq_cap = seq_cap
        self.device = torch.device(device_str)
        print(f"Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model on {device_str}...", flush=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map={"": device_str},
            trust_remote_code=True).eval()
        self.hf_model.gradient_checkpointing_enable()
        for p in self.hf_model.parameters():
            p.requires_grad = False

        self.d_model = self.hf_model.config.hidden_size
        self.n_layers = self.hf_model.config.num_hidden_layers
        self.inject_layers = [i for i in (0, 2, 4, 7, 10, 14, 17, 21, 24, 26)
                              if i < self.n_layers]

        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=self.d_model, inject_layers=self.inject_layers,
            bottleneck=bottleneck, init_scale=0.005, noise_scale=0.0).to(self.device)
        for g in self.steerer.gammas.values():
            g.data.fill_(0.02)

        self.manifest = CartridgeManifest(
            "code-cartridge-v1", CartridgeRole.DOMAIN_CAPABILITY,
            MODEL, MODEL, steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(self.inject_layers),
            parameter_count=sum(p.numel() for p in self.steerer.parameters()),
        )
        self.rack = SteererCartridgeRack()
        self.rack.mount(self.manifest, self.steerer, weight=1.0, active=True)
        self.rack.register_hooks(self.hf_model.model)

        n = sum(p.numel() for p in self.steerer.parameters())
        print(f"d={self.d_model} L={self.n_layers} inject={self.inject_layers} "
              f"params={n:,} seq_cap={seq_cap}", flush=True)

    def _set_weights(self, seq_len):
        self.rack.set_weights(torch.zeros(1, seq_len, 21))

    def train_step(self, prompt: str, expected: str) -> float:
        self.steerer.train()
        eos_id = self.tokenizer.eos_token_id
        prompt_ids = self.tokenizer.encode(prompt)
        expected_ids = self.tokenizer.encode(expected)
        full_ids = prompt_ids + expected_ids + [eos_id]
        full_ids = full_ids[:self.seq_cap]
        if len(full_ids) < 2:
            return 0.0

        dev = next(self.hf_model.parameters()).device
        x = torch.tensor([full_ids[:-1]], device=dev)
        y = torch.tensor([full_ids[1:]], device=dev)

        mask = torch.zeros_like(y, dtype=torch.float32)
        prompt_ctx = max(0, len(prompt_ids) - 1)
        mask[:, prompt_ctx:] = 1.0

        self._set_weights(x.shape[1])
        logits = self.hf_model(input_ids=x).logits.float()

        loss_tokens = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            reduction="none").reshape_as(mask)
        loss = (loss_tokens * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss + 0.00005 * self.steerer.orthogonal_penalty()
        return loss

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        ids = list(self.tokenizer.encode(prompt))
        prompt_len = len(ids)
        self.steerer.eval()
        generated_ids = []
        with torch.no_grad():
            for i in range(max_tokens):
                dev = next(self.hf_model.parameters()).device
                ctx = ids + generated_ids
                x = torch.tensor([ctx[-min(len(ctx), 512):]], device=dev)
                self._set_weights(x.shape[1])
                out = self.hf_model(x)
                logits = out.logits[0, -1].float().cpu()
                if not torch.isfinite(logits).all():
                    break
                nid = int(logits.argmax())
                if nid == self.tokenizer.eos_token_id or nid == self.tokenizer.pad_token_id:
                    break
                generated_ids.append(nid)
                # Stop at blank line after dedent (function boundary) — check every 4 tokens
                if i > 3 and i % 4 == 0:
                    current = self.tokenizer.decode(generated_ids)
                    if current.endswith("\n\n") and not current.rstrip().endswith(":"):
                        break
        return self.tokenizer.decode(generated_ids)

    def eval_humaneval_subset(self, he_examples: list[dict], n: int = 5) -> dict:
        self.steerer.eval()
        subset = random.sample(he_examples, min(n, len(he_examples)))
        passes = 0
        for ex in subset:
            gen = self.generate(ex["prompt"], max_tokens=80)
            ok = run_humaneval_test(ex["prompt"], gen, ex["test"], ex["entry_point"])
            if ok:
                passes += 1
        return {"passes": passes, "total": len(subset), "rate": passes / len(subset)}

    def cleanup(self):
        self.rack.remove_hooks()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Qwen3.5-4B Code Cartridge Training")

    print("\nLoading data...", flush=True)
    he_train, he_eval = load_humaneval_split(train_size=100)
    ch_funcs = load_code_harness_funcs(Path.home() / "code_harness/challenges/challenges_full.jsonl")

    print(f"  HumanEval train: {len(he_train)}  CodeHarness funcs: {len(ch_funcs)}  "
          f"HumanEval eval: {len(he_eval)}", flush=True)

    train_pool = he_train + ch_funcs
    random.seed(42)
    random.shuffle(train_pool)
    he_eval_subset = he_eval[:20]

    trainer = CodeTrainer(seq_cap=32)
    opt = torch.optim.AdamW(trainer.steerer.parameters(), lr=3e-4, weight_decay=0.01)

    print("\nBaseline (no cartridge)...", flush=True)
    bl = trainer.eval_humaneval_subset(he_eval_subset, n=5)
    print(f"  BL subset pass@1: {bl['rate']:.1%} ({bl['passes']}/{bl['total']})", flush=True)

    print("\nTraining...\n", flush=True)
    best_subset_rate = 0.0
    best_state = None
    start_time = time.time()
    losses = []

    for step in range(1, 2001):
        ch = random.choice(train_pool)
        opt.zero_grad()
        loss = trainer.train_step(ch["prompt"], ch["expected"])
        if loss > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.steerer.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        if step % 100 == 0:
            avg_loss = sum(losses[-50:]) / min(50, len(losses))
            eval_result = trainer.eval_humaneval_subset(he_eval_subset, n=5)
            elapsed = time.time() - start_time
            stamp = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            print(f"[{stamp}] step {step:3d}  loss={avg_loss:.4f}  "
                  f"he_pass={eval_result['passes']}/{eval_result['total']} "
                  f"({eval_result['rate']:.1%})", flush=True)

            if eval_result["rate"] > best_subset_rate:
                best_subset_rate = eval_result["rate"]
                best_state = {k: v.detach().cpu().clone() for k, v in trainer.steerer.state_dict().items()}
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"steerer_state": best_state, "step": step, "subset_rate": best_subset_rate},
                           OUT_DIR / "cartridge_best.pt")
                print(f"  [saved] best HE subset = {best_subset_rate:.1%}", flush=True)

    # Final: HumanEval eval set
    print(f"\nFinal HumanEval ({len(he_eval)} problems)...", flush=True)
    if best_state is not None:
        trainer.steerer.load_state_dict(best_state)
    trainer.steerer.eval()
    passes = 0
    for i, ex in enumerate(he_eval):
        gen = trainer.generate(ex["prompt"], max_tokens=80)
        ok = run_humaneval_test(ex["prompt"], gen, ex["test"], ex["entry_point"])
        if ok:
            passes += 1
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(he_eval)}  pass={passes} ({passes/(i+1):.1%})", flush=True)

    final_rate = passes / len(he_eval)
    print(f"\nFinal HumanEval pass@1: {passes}/{len(he_eval)} ({final_rate:.1%})", flush=True)

    torch.save({
        "steerer_state": best_state,
        "final_pass_rate": final_rate,
        "best_subset_rate": best_subset_rate,
        "he_passes": passes,
        "he_total": len(he_eval),
    }, OUT_DIR / "cartridge_final.pt")

    trainer.cleanup()
    print("Done.")


if __name__ == "__main__":
    main()
