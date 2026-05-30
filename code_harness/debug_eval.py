#!/usr/bin/env python3
"""Debug: verify HumanEval eval harness + check 7B Coder generation quality."""
import torch, tempfile, subprocess, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype=torch.float16, device_map={"": "cuda:0"}, trust_remote_code=True).eval()

he = load_dataset("openai_humaneval", split="test")

for idx in range(3):
    ex = he[idx]
    prompt = ex["prompt"]
    correct = ex["canonical_solution"]
    test_code = ex["test"]
    entry = ex["entry_point"]

    # Verify correct solution passes
    full = f"{prompt}{correct}\n\n{test_code}\n\ncheck({entry})"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full)
        tmp = f.name
    r = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=10)
    canonical_pass = r.returncode == 0
    os.unlink(tmp)

    # Generate
    ids = tk.encode(prompt)
    plen = len(ids)
    x = torch.tensor([ids]).cuda()
    with torch.no_grad():
        for _ in range(80):
            out = m(x)
            nid = int(out.logits[0, -1].argmax())
            if nid == tk.eos_token_id:
                break
            ids.append(nid)
            x = torch.tensor([ids[-512:]]).cuda()
    gen = tk.decode(ids[plen:])

    # Test generated
    full2 = f"{prompt}{gen}\n\n{test_code}\n\ncheck({entry})"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full2)
        tmp2 = f.name
    r2 = subprocess.run(["python3", tmp2], capture_output=True, text=True, timeout=10)
    gen_pass = r2.returncode == 0
    os.unlink(tmp2)

    print(f"[{idx}] {ex['task_id']}")
    print(f"  Canonical: {'PASS' if canonical_pass else 'FAIL'}")
    print(f"  Generated: {'PASS' if gen_pass else 'FAIL'}")
    print(f"  Gen ({len(gen)} chars): {repr(gen[:120])}")
    if r2.stderr:
        errs = [l for l in r2.stderr.strip().split("\n") if l.strip()]
        print(f"  Stderr: {errs[-1] if errs else 'none'}")
    print()
