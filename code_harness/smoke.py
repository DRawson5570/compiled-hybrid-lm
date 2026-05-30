#!/usr/bin/env python3
"""Minimal smoke test: Qwen3.5-4B + cartridge. Just verify gradient flows, loss drops, no NaN."""
import sys, json, random
from pathlib import Path
sys.path.insert(0, str(Path.home() / "deepseek_experiments"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from sandbox import run_test

MODEL = "Qwen/Qwen3.5-4B"
CHALLENGES = Path.home() / "code_harness/challenges/challenges_full.jsonl"


def load_challenges(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class SmokeTest:
    def __init__(self, device_str="cuda:0"):
        self.device_str = device_str
        print(f"Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model on {device_str}...", flush=True)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map={"": device_str},
            trust_remote_code=True).eval()
        self.hf_model.gradient_checkpointing_enable()
        for p in self.hf_model.parameters():
            p.requires_grad = False
        t1.record()
        torch.cuda.synchronize()
        print(f"  Loaded in {t0.elapsed_time(t1)/1000:.1f}s", flush=True)

        d = self.hf_model.config.hidden_size
        n_layers = self.hf_model.config.num_hidden_layers
        self.inject_layers = [i for i in (0, 2, 4, 7, 10, 14, 17, 21, 24, 26) if i < n_layers]

        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=d, inject_layers=self.inject_layers,
            bottleneck=128, init_scale=0.005, noise_scale=0.0)
        for g in self.steerer.gammas.values():
            g.data.fill_(0.02)

        self.manifest = CartridgeManifest(
            "smoke-test", CartridgeRole.DOMAIN_CAPABILITY,
            MODEL, MODEL, steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(self.inject_layers),
            parameter_count=sum(p.numel() for p in self.steerer.parameters()),
        )
        self.rack = SteererCartridgeRack()
        self.rack.mount(self.manifest, self.steerer, weight=1.0, active=True)
        self.rack.register_hooks(self.hf_model.model)

        n = sum(p.numel() for p in self.steerer.parameters())
        print(f"Cartridge: {n:,} params, inject={self.inject_layers}", flush=True)

    def train_step(self, prompt, expected):
        self.steerer.train()
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)
        full_ids = self.tokenizer.encode(prompt + expected)[:64]
        if len(full_ids) < 2:
            return None

        dev = next(self.hf_model.parameters()).device
        x = torch.tensor([full_ids[:-1]], device=dev)
        y = torch.tensor([full_ids[1:]], device=dev)

        mask = torch.zeros_like(y, dtype=torch.float32)
        mask[:, max(0, prompt_len - 1):] = 1.0

        self.rack.set_weights(torch.zeros(1, x.shape[1], 21))
        logits = self.hf_model(input_ids=x).logits.float()

        loss_tokens = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            reduction="none").reshape_as(mask)
        loss = (loss_tokens * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss + 0.00005 * self.steerer.orthogonal_penalty()
        return loss

    def generate(self, prompt, max_tokens=20):
        ids = list(self.tokenizer.encode(prompt))
        prompt_len = len(ids)
        self.steerer.eval()
        with torch.no_grad():
            for _ in range(max_tokens):
                dev = next(self.hf_model.parameters()).device
                x = torch.tensor([ids[-256:]], device=dev)
                self.rack.set_weights(torch.zeros(1, x.shape[1], 21))
                out = self.hf_model(x)
                logits = out.logits[0, -1].float().cpu()
                if not torch.isfinite(logits).all():
                    break
                nid = int(logits.argmax())
                if nid == self.tokenizer.eos_token_id:
                    break
                ids.append(nid)
        return self.tokenizer.decode(ids[prompt_len:])

    def cleanup(self):
        self.rack.remove_hooks()


def main():
    challenges = load_challenges(CHALLENGES)
    tier1 = [c for c in challenges if c["tier"] in (1, 4)]
    random.seed(42)
    random.shuffle(tier1)
    train = tier1[:3]   # just 3 examples
    test = tier1[3:4]   # 1 held-out

    print(f"\nTrain: {[c['title'] for c in train]}")
    print(f"Test: {[c['title'] for c in test]}")

    t = SmokeTest()

    # 1. Baseline generation
    print("\n--- Baseline (no cartridge) ---")
    for ch in test:
        gen = t.generate(ch["prompt"], max_tokens=20)
        r = run_test(gen, ch["test_code"])
        print(f"  {ch['title']}: gen={gen[:40].strip()!r}  pass={r.passed}")
    baseline_pass = 0  # track if we get any pass

    # 2. Training sanity: check gradients flow
    print("\n--- Training sanity ---")
    opt = torch.optim.AdamW(t.steerer.parameters(), lr=1e-3)

    loss0 = None
    for step in range(1, 11):
        ch = random.choice(train)
        opt.zero_grad()
        loss = t.train_step(ch["prompt"], ch["expected"])
        if loss is None:
            continue
        loss.backward()

        # Check gradients are non-zero
        total_grad = sum(p.grad.abs().sum().item() for p in t.steerer.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(t.steerer.parameters(), 1.0)
        opt.step()

        if loss0 is None:
            loss0 = loss.item()

        print(f"  step {step:2d}  loss={loss.item():.4f}  grad_sum={total_grad:.2e}  "
              f"loss_delta={loss.item()-loss0:+.4f}", flush=True)

    # 3. Cartridge evaluation
    print("\n--- Cartridge ---")
    for ch in test:
        gen = t.generate(ch["prompt"], max_tokens=20)
        r = run_test(gen, ch["test_code"])
        print(f"  {ch['title']}: gen={gen[:40].strip()!r}  pass={r.passed}")

    # 4. Check for NaN
    for name, p in t.steerer.named_parameters():
        if torch.isnan(p).any():
            print(f"  NAN in {name}!")
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"  NAN grad in {name}!")

    t.cleanup()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
