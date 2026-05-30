#!/usr/bin/env python3
"""Rejection-sampling fine-tuning (RFT / STaR) of a tiny adapter cartridge.

Why this and not autoregressive CE on canonical solutions:
  - CE on canonical code is OFF-policy: it optimises P(canonical | gold prefix).
    At inference the model conditions on its OWN prefix -> exposure bias ->
    compounding errors -> the famous "ppl 1.15 but 0 pass" failure.
  - RFT trains the cartridge on the model's OWN passing rollouts (on-policy),
    so the training distribution matches deployment. This is the objective
    that actually correlates with pass@1.

Pipeline:
  1. Load passing (prompt, code) pairs from a candidate cache (gen_candidates.py).
  2. Masked CE on the assistant code block only, gradients flow ONLY to the
     frozen-model adapter cartridge (~few M params).
  3. Periodically eval greedy pass@1 on held-out HumanEval with the cartridge
     ACTIVE vs the base model (rack deactivated). Save the best cartridge.
"""
import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "deepseek_experiments"))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack

MODEL = "Qwen/Qwen3.5-4B"

MBPP_INSTRUCTION = (
    "{nl}\n\nYour code must pass these tests:\n{tests}\n\n"
    "Return ONLY the function inside a single ```python code block, no explanation."
)
HE_INSTRUCTION = (
    "Complete the following Python function. Return ONLY the complete function "
    "(including the signature) inside a single ```python code block, no explanation.\n\n"
    "```python\n{prompt}\n```"
)


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def run_program(program: str, timeout: float = 8.0) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
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


def build_he_program(code, prob):
    code = code.strip("\n")
    prog = code if f"def {prob['entry_point']}" in code else f"{prob['prompt']}{code}"
    return f"{prog}\n\n{prob['test']}\n\ncheck({prob['entry_point']})\n"


class RFTTrainer:
    def __init__(self, device="cuda:0", bottleneck=128, seq_cap=640, load_4bit=True):
        self.device = device
        self.seq_cap = seq_cap
        self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        if load_4bit:
            # Frozen 4-bit NF4 base (QLoRA pattern): ~3.1GB vs 8.46GB fp16, so the
            # 4B model + trainable fp32 adapter fit comfortably on a 10GB 3080.
            qconf = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL, quantization_config=qconf, device_map={"": device},
                dtype=torch.float16, trust_remote_code=True).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.float16, device_map={"": device}, trust_remote_code=True).eval()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad = False

        d = self.model.config.hidden_size
        n = self.model.config.num_hidden_layers
        self.inject_layers = [i for i in (0, 2, 4, 7, 10, 14, 17, 21, 24, 26, 28, 30) if i < n]
        self.steerer = FeatureConditionedAdapterSteerer(
            d_model=d, inject_layers=self.inject_layers, bottleneck=bottleneck,
            init_scale=0.005, noise_scale=0.0, semantic_dim=16).to(device)
        for g in self.steerer.gammas.values():
            g.data.fill_(0.02)
        # adapter params in fp32 for stable optimisation
        self.steerer.float()

        self.manifest = CartridgeManifest(
            "code-rft-v1", CartridgeRole.DOMAIN_CAPABILITY, MODEL, MODEL,
            steerer_class="FeatureConditionedAdapterSteerer",
            inject_layers=tuple(self.inject_layers),
            parameter_count=sum(p.numel() for p in self.steerer.parameters()))
        self.rack = SteererCartridgeRack()
        self.rack.mount(self.manifest, self.steerer, weight=1.0, active=True)
        self.rack.register_hooks(self.model.model)
        nparam = sum(p.numel() for p in self.steerer.parameters())
        print(f"d={d} L={n} inject={self.inject_layers} adapter_params={nparam:,}", flush=True)

    def set_active(self, on: bool):
        self.rack.activate(self.manifest.cartridge_id, on)

    def chat_prefix(self, user_content: str) -> str:
        msgs = [{"role": "user", "content": user_content}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def train_step(self, user_content: str, code: str) -> torch.Tensor | None:
        self.steerer.train()
        # CRITICAL: the base must be in train() mode for gradient checkpointing to
        # engage -- each qwen3_5 decoder layer gates on `self.gradient_checkpointing
        # and self.training`. In eval() the GC flag is silently ignored and the full
        # linear-attention autograd graph is retained (~8.2GB @ seq256, OOMs >256 on
        # a 10GB card). train() halves activation memory (5.34GB @ seq512). Params
        # stay frozen (requires_grad=False) and qwen3_5 has zero dropout, so the
        # frozen base's outputs are identical to eval() mode.
        self.model.train()
        prefix = self.chat_prefix(user_content)
        target = f"```python\n{code.strip()}\n```"
        prefix_ids = self.tok.encode(prefix)
        target_ids = self.tok.encode(target, add_special_tokens=False) + [self.tok.eos_token_id]
        full = (prefix_ids + target_ids)[: self.seq_cap]
        if len(full) < 4 or len(prefix_ids) >= len(full) - 1:
            return None
        dev = self.device
        x = torch.tensor([full[:-1]], device=dev)
        y = torch.tensor([full[1:]], device=dev)
        # Only the target (code) tokens are supervised. Run the inner model and
        # apply lm_head ONLY to the supervised positions so we never materialise
        # full-sequence float logits (the dominant activation on a 10GB GPU).
        start = len(prefix_ids) - 1
        hidden = self.model.model(input_ids=x).last_hidden_state[:, start:, :]
        logits = self.model.lm_head(hidden).float()
        y_t = y[:, start:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_t.reshape(-1))
        return loss + 0.00005 * self.steerer.orthogonal_penalty()

    @torch.no_grad()
    def gen_batch(self, prompts, max_new=320, batch=16):
        self.steerer.eval()
        self.model.eval()
        # GC is incompatible with the KV cache -> disable it (and re-enable cache)
        # during generation, otherwise generate runs cache-less and is O(n^2) slow.
        self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        torch.cuda.empty_cache()
        outs = []
        for b in range(0, len(prompts), batch):
            chunk = prompts[b:b + batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True).to(self.device)
            o = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=self.tok.eos_token_id)
            gen = o[:, enc["input_ids"].shape[1]:]
            outs.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
            del enc, o, gen
        # restore training config
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        gc.collect()
        torch.cuda.empty_cache()
        return outs

    def eval_humaneval(self, he_probs, active: bool, max_new=320, batch=4) -> float:
        self.set_active(active)
        prompts = [self.chat_prefix(HE_INSTRUCTION.format(prompt=p["prompt"])) for p in he_probs]
        gens = self.gen_batch(prompts, max_new=max_new, batch=batch)
        passes = 0
        for p, g in zip(he_probs, gens):
            if run_program(build_he_program(extract_code(g), p)):
                passes += 1
        return passes / len(he_probs)


def load_passing(cache_path, max_per_problem=2):
    pairs = []
    with open(cache_path) as f:
        for line in f:
            r = json.loads(line)
            passing = [c["code"] for c in r["candidates"] if c["passed"]]
            random.shuffle(passing)
            for code in passing[:max_per_problem]:
                user = MBPP_INSTRUCTION.format(nl=r["nl"], tests="\n".join(r["tests"]))
                pairs.append((user, code))
    return pairs


def humaneval_problems():
    return [{"task_id": e["task_id"], "prompt": e["prompt"], "test": e["test"],
             "entry_point": e["entry_point"]} for e in load_dataset("openai/openai_humaneval", split="test")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path.home() / "code_harness/cand_mbpp_test.jsonl"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=50)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--seq-cap", type=int, default=512)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--out", default=str(Path.home() / "deepseek_experiments/artifacts/qwen35_4b_rft"))
    args = ap.parse_args()
    random.seed(42)

    pairs = load_passing(args.cache, args.max_per_problem)
    print(f"Loaded {len(pairs)} passing (prompt,code) training pairs from {args.cache}", flush=True)
    he = humaneval_problems()
    random.shuffle(he)
    he_eval = he[: args.eval_n]

    t = RFTTrainer(device=args.device, seq_cap=args.seq_cap)
    opt = torch.optim.AdamW(t.steerer.parameters(), lr=args.lr, weight_decay=0.01)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nBaseline greedy pass@1 (cartridge OFF)...", flush=True)
    base_rate = t.eval_humaneval(he_eval, active=False, batch=args.eval_batch)
    print(f"  BASE pass@1 ({args.eval_n}): {base_rate:.1%}", flush=True)
    cart0 = t.eval_humaneval(he_eval, active=True, batch=args.eval_batch)
    print(f"  CARTRIDGE@init pass@1 ({args.eval_n}): {cart0:.1%}", flush=True)

    best = cart0
    losses = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        user, code = random.choice(pairs)
        opt.zero_grad()
        loss = t.train_step(user, code)
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(t.steerer.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        if step % 50 == 0:
            avg = sum(losses[-50:]) / max(1, len(losses[-50:]))
            el = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
            print(f"[{el}] step {step} loss={avg:.4f}", flush=True)
        if step % args.eval_every == 0:
            rate = t.eval_humaneval(he_eval, active=True, batch=args.eval_batch)
            print(f"  >>> step {step} CARTRIDGE pass@1={rate:.1%}  (base {base_rate:.1%})", flush=True)
            if rate >= best:
                best = rate
                torch.save({"steerer_state": {k: v.detach().cpu().clone()
                                              for k, v in t.steerer.state_dict().items()},
                            "step": step, "rate": rate, "base_rate": base_rate,
                            "inject_layers": t.inject_layers},
                           out_dir / "cartridge_best.pt")
                print(f"  [saved] best={best:.1%}", flush=True)

    print(f"\nDONE. base={base_rate:.1%} best_cartridge={best:.1%}", flush=True)


if __name__ == "__main__":
    main()
