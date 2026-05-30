#!/usr/bin/env python3
"""Smoke test: Qwen3.5-4B + code cartridge PoC.
Trains with masked CE on correct completions (autoregressive, no RL loop).
20 examples, 5 held out, 100 steps. Verifies gradient flow, loss drop, no regression.
"""
import sys, json, random, time, os
from pathlib import Path
sys.path.insert(0, str(Path.home() / "deepseek_experiments"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from sandbox import run_test

MODEL = "Qwen/Qwen3.5-4B"
OUT_DIR = Path.home() / "deepseek_experiments/artifacts/qwen35_4b_code_cartridge_poc"
CHALLENGES = Path.home() / "code_harness/challenges/challenges_full.jsonl"


def load_challenges(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class CodeCartridgePoC:
    def __init__(self, model_name=MODEL, device_str="cuda:0", bottleneck=128):
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading {model_name}...", flush=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map={"": self.device_str},
            trust_remote_code=True).eval()
        self.hf_model.gradient_checkpointing_enable()
        for p in self.hf_model.parameters():
            p.requires_grad = False

        d = self.hf_model.config.hidden_size
        n_layers = self.hf_model.config.num_hidden_layers
        self.inject_layers = [i for i in (0, 2, 4, 7, 10, 14, 17, 21, 24, 26) if i < n_layers]
        print(f"d_model={d}, layers={n_layers}, inject_layers={self.inject_layers}", flush=True)

        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=d, inject_layers=self.inject_layers,
            bottleneck=bottleneck, init_scale=0.005, noise_scale=0.0)
        for g in self.steerer.gammas.values():
            g.data.fill_(0.02)

        self.manifest = CartridgeManifest(
            "code-cartridge-poc", CartridgeRole.DOMAIN_CAPABILITY,
            model_name, model_name, steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(self.inject_layers),
            parameter_count=sum(p.numel() for p in self.steerer.parameters()),
        )
        self.rack = SteererCartridgeRack()
        self.rack.mount(self.manifest, self.steerer, weight=1.0, active=False)
        self.rack.register_hooks(self.hf_model.model)
        self.steerer_enabled = False

        self.n_params = sum(p.numel() for p in self.steerer.parameters())
        print(f"Cartridge params: {self.n_params:,}", flush=True)

    def set_enabled(self, enabled: bool):
        self.steerer_enabled = enabled
        self.rack.activate(self.manifest.cartridge_id, enabled)

    def forward_with_cartridge(self, input_ids, attention_mask=None):
        self.rack.set_weights(torch.zeros(1, input_ids.shape[1], 21))
        out = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.float()

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        ids = list(self.tokenizer.encode(prompt))
        prompt_len = len(ids)
        with torch.no_grad():
            for _ in range(max_tokens):
                x = torch.tensor([ids[-1024:]])
                model_device = next(self.hf_model.parameters()).device
                x = x.to(model_device)
                if self.steerer_enabled:
                    self.rack.set_weights(torch.zeros(1, x.shape[1], 21))
                else:
                    self.rack.activate(self.manifest.cartridge_id, False)
                out = self.hf_model(x)
                logits = out.logits[0, -1].float().cpu()
                if not torch.isfinite(logits).all():
                    break
                nid = int(logits.argmax())
                if nid == self.tokenizer.eos_token_id:
                    break
                ids.append(nid)
        return self.tokenizer.decode(ids[prompt_len:])

    def compute_eval_accuracy(self, eval_challenges: list[dict]) -> tuple[int, int]:
        """Run evaluation with cartridge ON. Returns (passes, total)."""
        self.steerer.eval()
        self.set_enabled(True)
        passes = 0
        for ch in eval_challenges:
            gen = self.generate(ch["prompt"], max_tokens=100)
            result = run_test(gen, ch["test_code"])
            if result.passed:
                passes += 1
        self.set_enabled(False)
        return passes, len(eval_challenges)

    def compute_baseline_accuracy(self, eval_challenges: list[dict]) -> tuple[int, int]:
        """Baseline: frozen model without cartridge."""
        self.set_enabled(False)
        passes = 0
        for ch in eval_challenges:
            gen = self.generate(ch["prompt"], max_tokens=100)
            result = run_test(gen, ch["test_code"])
            if result.passed:
                passes += 1
        return passes, len(eval_challenges)

    def train_step(self, prompt: str, expected: str) -> float:
        """Masked cross-entropy on the correct completion (autoregressive)."""
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)

        full_text = prompt + expected
        full_ids = self.tokenizer.encode(full_text)[:128]
        if len(full_ids) < 2:
            return 0.0

        model_device = next(self.hf_model.parameters()).device
        x = torch.tensor([full_ids[:-1]], device=model_device)
        y = torch.tensor([full_ids[1:]], device=model_device)

        mask = torch.zeros_like(y, dtype=torch.float32)
        mask[:, max(0, prompt_len - 1):] = 1.0

        logits = self.forward_with_cartridge(x)
        loss_tokens = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            reduction="none").reshape_as(mask)
        loss = (loss_tokens * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss + 0.00005 * self.steerer.orthogonal_penalty()
        return loss

    def cleanup(self):
        self.rack.remove_hooks()


def main():
    challenges = load_challenges(CHALLENGES)
    tier1 = [c for c in challenges if c["tier"] in (1, 4)]
    print(f"Loaded {len(challenges)} challenges ({len(tier1)} tier-1/4)")

    random.seed(42)
    random.shuffle(tier1)
    pool = tier1[:20]
    train_pool = pool[:15]
    eval_pool = pool[15:20]
    print(f"Train: {len(train_pool)}, Eval (held-out): {len(eval_pool)}")
    for i, c in enumerate(eval_pool):
        print(f"  eval[{i}]: {c['title']}")

    trainer = CodeCartridgePoC()
    opt = torch.optim.AdamW(trainer.steerer.parameters(), lr=3e-4, weight_decay=0.01)

    baseline_pass, baseline_total = trainer.compute_baseline_accuracy(eval_pool)
    print(f"\nBaseline (no cartridge): {baseline_pass}/{baseline_total} pass", flush=True)

    best_eval_pass = 0
    trainer.steerer.train()

    for step in range(1, 101):
        ch = random.choice(train_pool)
        opt.zero_grad()
        trainer.set_enabled(True)
        loss = trainer.train_step(ch["prompt"], ch["expected"])
        if loss > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.steerer.parameters(), 1.0)
            opt.step()

        if step % 25 == 0:
            trainer.steerer.eval()
            eval_pass, eval_total = trainer.compute_eval_accuracy(eval_pool)
            trainer.steerer.train()
            delta = eval_pass - baseline_pass
            tag = "OK" if eval_pass >= baseline_pass else "REGRESS"
            print(f"[step {step:3d}] loss={loss:.4f} eval={eval_pass}/{eval_total} "
                  f"(baseline={baseline_pass}/{baseline_total}, Δ={delta:+d}) {tag}", flush=True)

            if eval_pass > best_eval_pass:
                best_eval_pass = eval_pass

    # Final
    trainer.steerer.eval()
    final_pass, final_total = trainer.compute_eval_accuracy(eval_pool)
    print(f"\nFinal: {final_pass}/{final_total} (baseline: {baseline_pass}/{baseline_total})")
    if final_pass >= baseline_pass:
        print("PASS: No regression.", flush=True)
    else:
        print(f"REGRESSION: lost {baseline_pass - final_pass} points", flush=True)

    for i, c in enumerate(eval_pool):
        gen = trainer.generate(c["prompt"], max_tokens=100)
        result = run_test(gen, c["test_code"])
        print(f"  [{i}] {c['title']}: {'PASS' if result.passed else 'FAIL'}")
        print(f"       prompt: {c['prompt'][:60].rstrip()}...")
        print(f"       gen:    {gen[:80].rstrip()}...")

    trainer.cleanup()


if __name__ == "__main__":
    main()
