#!/usr/bin/env python3
"""Quick test: Qwen2.5-Coder-1.5B-Instruct HumanEval pass@1 on first problem."""
import torch, tempfile, subprocess, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct",
    torch_dtype=torch.float16, device_map={"": "cuda:0"}, trust_remote_code=True).eval()

he = load_dataset("openai_humaneval", split="test")
ex = he[0]
print(f"Keys: {list(ex.keys())}")
prompt = ex["prompt"]
solution = ex["canonical_solution"]
print(f"Prompt ({len(prompt)} chars):\n{prompt[:200]}")
print(f"\nExpected ({len(solution)} chars):\n{solution[:200]}")

ids = tk.encode(prompt)
prompt_len = len(ids)
x = torch.tensor([ids]).cuda()
with torch.no_grad():
    for _ in range(80):
        out = m(x)
        nid = int(out.logits[0, -1].argmax())
        if nid == tk.eos_token_id or nid == tk.pad_token_id:
            break
        ids.append(nid)
        x = torch.tensor([ids[-512:]]).cuda()
gen = tk.decode(ids[prompt_len:])
print(f"\nGenerated ({len(gen)} chars):\n{gen[:300]}")

entry = ex["entry_point"]
test = ex["test"]
full_code = f"{prompt}{gen}\n\n{test}\n\ncheck({entry})"
with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
    f.write(full_code)
    tmp = f.name
r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=10)
try:
    os.unlink(tmp)
except OSError:
    pass
print(f"\nTest result: {'PASS' if r.returncode == 0 else 'FAIL'} (rc={r.returncode})")
if r.stderr:
    err_lines = [l for l in r.stderr.split("\n") if l.strip()]
    print(f"Stderr: {err_lines[-1] if err_lines else 'none'}")
