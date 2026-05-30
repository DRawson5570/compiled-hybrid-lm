#!/usr/bin/env python3
"""Quick verify: zero layers 14+21, run full HumanEval A/B."""
import json, re, subprocess, sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path.home() / "deepseek_experiments"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from hybrid.superposition_steerer_v3 import FeatureConditionedAdapterSteerer
from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack

MODEL = "Qwen/Qwen3.5-4B"
CKPT = Path.home() / "deepseek_experiments/artifacts/qwen35_4b_rft/cartridge_best.pt"
INSTRUCTION = (
    "Complete the following Python function. Return ONLY the complete function "
    "(including the signature) inside a single ```python code block, no explanation.\n\n"
    "```python\n{prompt}\n```"
)

def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text

def run_test(program, prob):
    full = f"{program}\n\n{prob['test']}\n\ncheck({prob['entry_point']})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full); tmp = f.name
    try:
        r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=10.0)
        return r.returncode == 0
    except: return False
    finally:
        try: os.unlink(tmp)
        except OSError: pass

ckpt = torch.load(CKPT, map_location="cpu")
inject_layers = ckpt["inject_layers"]

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map={"": "cuda:0"}, trust_remote_code=True).eval()

s = FeatureConditionedAdapterSteerer(d_model=m.config.hidden_size, inject_layers=inject_layers,
    bottleneck=128, init_scale=0.005, noise_scale=0.0, semantic_dim=16).to("cuda:0").float()
s.load_state_dict(ckpt["steerer_state"])
s.eval()

# Zero layers 14 and 21
s.gammas["14"].data.fill_(0.0)
s.gammas["21"].data.fill_(0.0)
print(f"Zeroed layers 14, 21. Other gammas:")
for k, g in sorted(s.gammas.items()):
    print(f"  {k}: {g.item():+.4f}")

manifest = CartridgeManifest("verify", CartridgeRole.DOMAIN_CAPABILITY, MODEL, MODEL,
    steerer_class="FeatureConditionedAdapterSteerer", inject_layers=tuple(inject_layers),
    parameter_count=sum(p.numel() for p in s.parameters()))
rack = SteererCartridgeRack()
rack.mount(manifest, s, weight=1.0, active=True)
rack.register_hooks(m.model)

probs = [{"task_id": e["task_id"], "prompt": e["prompt"], "test": e["test"], "entry_point": e["entry_point"]}
         for e in load_dataset("openai_humaneval", split="test")]

def chat(prompt):
    msgs = [{"role": "user", "content": INSTRUCTION.format(prompt=prompt)}]
    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def evaluate(active, label):
    rack.activate(manifest.cartridge_id, active)
    passed = 0
    for i, p in enumerate(probs):
        enc = tok([chat(p["prompt"])], return_tensors="pt").to("cuda:0")
        o = m.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=tok.eos_token_id)
        code = extract_code(tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        program = code if f"def {p['entry_point']}" in code else f"{p['prompt']}{code}"
        if run_test(program, p): passed += 1
        if (i+1) % 40 == 0:
            print(f"  [{label}] {i+1}/{len(probs)} pass={passed} ({passed/(i+1):.1%})", flush=True)
    print(f"  [{label}] FINAL: {passed}/{len(probs)} = {passed/len(probs):.1%}", flush=True)
    return passed

print(f"\nEvaluating on {len(probs)} HumanEval problems...\n")

base = evaluate(active=False, label="BASE")
print()
cart = evaluate(active=True, label="CART (zero 14+21)")
delta = cart - base
print(f"\nDelta: {delta:+d} ({delta/len(probs):+.1%})")

rack.remove_hooks()
