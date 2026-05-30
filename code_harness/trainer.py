#!/usr/bin/env python3
"""Code cartridge trainer with RL feedback loop."""
import sys, json, random, time, os
from pathlib import Path
sys.path.insert(0, str(Path.home() / "deepseek_experiments"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from sandbox import run_test, TestResult

MODEL = "Qwen/Qwen2.5-3B"
OUT_DIR = Path.home() / "deepseek_experiments/artifacts/qwen25_3b_code_cartridge"
CHALLENGES = Path.home() / "code_harness/challenges/challenges_full.jsonl"


def load_challenges(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class CodeCartridgeTrainer:
    def __init__(self, model_name=MODEL, device="cuda", bottleneck=128):
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()
        for p in self.hf_model.parameters():
            p.requires_grad = False

        d = self.hf_model.config.hidden_size
        layers = len(self.hf_model.model.layers)
        self.inject_layers = [i for i in (0,2,4,7,10,14,17,21,24,26) if i < layers]
        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=d, inject_layers=self.inject_layers,
            bottleneck=bottleneck, init_scale=0.005, noise_scale=0.0).to(device)
        for g in self.steerer.gammas.values():
            g.data.fill_(0.02)

        self.manifest = CartridgeManifest(
            "code-cartridge", CartridgeRole.DOMAIN_CAPABILITY,
            model_name, model_name, steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(self.inject_layers),
            parameter_count=sum(p.numel() for p in self.steerer.parameters()),
        )
        self.rack = SteererCartridgeRack()
        self.rack.mount(self.manifest, self.steerer, weight=1.0, active=False)
        self.rack.register_hooks(self.hf_model.model)
        self.enabled = False

    def set_enabled(self, e):
        self.enabled = e
        self.rack.activate(self.manifest.cartridge_id, e)

    def set_weights(self, seq_len):
        self.rack.set_weights(self.torch.zeros(1, seq_len, 21, device=self.device))

    def generate(self, prompt, max_tokens=100):
        ids = list(self.tokenizer.encode(prompt))
        model_device = next(self.hf_model.parameters()).device
        with self.torch.no_grad():
            for _ in range(max_tokens):
                x = self.torch.tensor([ids[-1024:]], device=model_device)
                if self.enabled:
                    self.set_weights(x.shape[1])
                else:
                    self.rack.activate(self.manifest.cartridge_id, False)
                out = self.hf_model(x)
                logits = out.logits[0, -1].float().cpu()
                if not self.torch.isfinite(logits).all():
                    break
                nid = int(logits.argmax())
                if nid == self.tokenizer.eos_token_id:
                    break
                ids.append(nid)
        return self.tokenizer.decode(ids[len(self.tokenizer.encode(prompt)):])

    def train_step(self, challenge: dict) -> tuple[float, TestResult]:
        """One training step: generate code, test, backprop loss."""
        prompt = challenge["prompt"]
        expected = challenge["expected"]
        test_code = challenge["test_code"]

        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)

        # Generate with cartridge ON
        self.set_enabled(True)
        self.steerer.train()

        generated = self.generate(prompt, max_tokens=200)
        result = run_test(generated, test_code)

        if not result.passed:
            full_text = prompt + expected
            full_ids = self.tokenizer.encode(full_text)[:512]
            model_device = next(self.hf_model.parameters()).device
            x = self.torch.tensor([full_ids[:-1]], device=model_device)
            y = self.torch.tensor([full_ids[1:]], device=model_device)
            mask = self.torch.zeros_like(y, dtype=self.torch.float32)
            mask[:, max(0, prompt_len - 1):] = 1.0
            self.set_weights(x.shape[1])

            logits = self.hf_model(x).logits.float()
            loss_tokens = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                reduction="none").reshape_as(mask)
            loss = (loss_tokens * mask).sum() / mask.sum().clamp(min=1.0)
            loss = loss + 0.00005 * self.steerer.orthogonal_penalty()
            loss.backward()
            return float(loss.item()), result

        return 0.0, result

    def cleanup(self):
        self.rack.remove_hooks()


def main():
    challenges = load_challenges(CHALLENGES)
    random.shuffle(challenges)
    tier1 = [c for c in challenges if c["tier"] == 1 or c["tier"] == 4]
    print(f"Loaded {len(challenges)} challenges ({len(tier1)} trainable)")

    trainer = CodeCartridgeTrainer(bottleneck=128)
    print(f"d_model={trainer.hf_model.config.hidden_size}, layers={len(trainer.hf_model.model.layers)}")
    print(f"steerer: {sum(p.numel() for p in trainer.steerer.parameters()):,} params")

    opt = torch.optim.AdamW(trainer.steerer.parameters(), lr=3e-4, weight_decay=0.01)
    best_pass = 0
    history = []

    for step in range(1, 2001):
        ch = random.choice(tier1[:100])  # training subset
        opt.zero_grad()
        loss, result = trainer.train_step(ch)
        if loss > 0:
            torch.nn.utils.clip_grad_norm_(trainer.steerer.parameters(), 1.0)
            opt.step()

        if result.passed:
            best_pass += 1

        if step % 50 == 0:
            trainer.steerer.eval()
            all_tier1 = [c for c in challenges if c["tier"] in (1, 4)]
            eval_challenges = all_tier1[-10:] if len(all_tier1) > 10 else all_tier1[-5:]
            eval_passes = 0
            for ch_eval in eval_challenges:
                gen = trainer.generate(ch_eval["prompt"], max_tokens=60)
                r = run_test(gen, ch_eval["test_code"])
                if r.passed:
                    eval_passes += 1
            acc = eval_passes / max(10, 1) * 100
            trainer.steerer.train()
            print(f"[step {step:3d}] loss={loss:.4f} train_pass={best_pass} eval_pass={eval_passes}/10 acc={acc:.0f}%", flush=True)
            history.append({"step": step, "loss": loss, "train_pass": best_pass, "eval_pass": eval_passes, "acc": acc})

            if eval_passes > best_pass:
                best_pass = eval_passes
                best_state = {k: v.detach().cpu().clone() for k, v in trainer.steerer.state_dict().items()}
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save({"steerer_state": best_state, "history": history}, OUT_DIR / "cartridge_best.pt")
                print(f"  [saved] eval_pass={eval_passes}", flush=True)

    trainer.cleanup()
    print(f"\nBest eval pass: {best_pass}/10")


if __name__ == "__main__":
    main()
