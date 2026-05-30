#!/usr/bin/env python3
import json, ast, random, glob
from pathlib import Path

REPO = Path("/home/drawson/deepseek_experiments")
OUT = Path("/home/drawson/code_harness/challenges/challenges_full.jsonl")
C = []
random.seed(42)

def add(tier, cat, prompt, expected, test, src, title=""):
    C.append({"id": f"{cat}_{len(C):04d}", "tier": tier, "category": cat,
              "prompt": prompt, "expected": expected,
              "test_code": test, "source_file": str(src), "title": title})

# CODE: Scan only hybrid/, experiments/, tests/ - LIMIT per directory
for scan_dir, max_per_dir in [("hybrid", 80), ("experiments", 40), ("tests", 30)]:
    funcs = []
    for py in sorted(REPO.glob(f"{scan_dir}/**/*.py")):
        rid = py.relative_to(REPO)
        if "archive" in str(rid) or "__pycache__" in str(rid): continue
        try: src = py.read_text()
        except: continue
        try: tree = ast.parse(src)
        except: continue
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
            if len(n.body) < 3: continue
            funcs.append((py, n, src))
    random.shuffle(funcs)
    for py, n, src_text in funcs[:max_per_dir]:
        lines = src_text.split("\n")
        bs, be = n.body[0].lineno - 1, n.body[-1].end_lineno
        if be - bs < 2: continue
        preamble = lines[max(0, n.lineno - 4):bs]
        prompt = "\n".join(preamble) + "\n"
        expected = "\n".join(lines[bs:be]) + "\n"
        test = f'import ast,sys\ncode=sys.stdin.read()\ntry:\n t=ast.parse(code)\n f=[x.name for x in ast.walk(t) if isinstance(x,ast.FunctionDef)]\n assert "{n.name}" in f,f"Missing {n.name}"\n print("PASS")\nexcept SyntaxError as e:print(f"FAIL:syntax {{e}}")\nexcept AssertionError as e:print(f"FAIL: {{e}}")'
        add(1, "code", prompt, expected, test, rid, f"{scan_dir}/{n.name}")

# FILE CHUNKS: Small files split into 2-3 chunks
for pat in ["hybrid/*.py", "experiments/*.py"]:
    for fp in sorted(REPO.glob(pat)):
        if fp.stat().st_size < 800 or fp.stat().st_size > 8000: continue
        src = fp.read_text(); lines = src.split("\n"); n = len(lines)
        for chunk_start in [n//3, n//2]:
            prompt = "\n".join(lines[:chunk_start]) + "\n"
            expected = "\n".join(lines[chunk_start:chunk_start + n//5]) + "\n"
            if len(prompt) < 30 or len(expected) < 15: continue
            test = "import ast,sys\ntry:\n ast.parse(sys.stdin.read());print('PASS')\nexcept SyntaxError as e:print(f'FAIL:{e}')"
            add(2, "file_chunk", prompt, expected, test, fp.relative_to(REPO))

# SHELL: All .sh files
sh_files = list(REPO.glob("*.sh")) + list(REPO.glob("scripts/*.sh"))
for sh in sh_files[:10]:
    src = sh.read_text(); lines = src.split("\n")
    mid = max(len(lines)//2, 5)
    prompt = "\n".join(lines[:mid]) + "\n"
    expected = "\n".join(lines[mid:min(mid+20, len(lines))]) + "\n"
    test = "import sys;c=sys.stdin.read();assert len(c)>15;print('PASS')"
    add(3, "shell", prompt, expected, test, sh.name)

# THEORY: Deep Q&A on every concept
THEORY = [
    ("AdamW decoupled weight decay", "Why does AdamW decouple weight decay from gradient-based updates? Why does this matter for our 21-channel steerer with vastly different gradient scales across channels?",
     "AdamW applies weight decay directly: w -= lr * (adam_update + wd * w). This makes weight decay uniform regardless of gradient magnitude. In our 21-channel steerer, local n-gram channels have gradients ~100x larger than global topic channels. L2-in-Adam would let large-gradient channels dodge regularization. AdamW ensures ALL channels receive equal decay."),
    ("Hook gradient flow through frozen model", "A frozen model (requires_grad=False) has steerer forward hooks registered. Does loss.backward() compute steerer gradients? Why?",
     "Yes. The model's forward pass creates activations in the autograd graph even though the parameters producing them don't require grad. The steerer computes deltas from these activations: delta = up(GELU(down(norm(h)) + feature(channels))). Since h participates in the graph and steerer params require grad, backprop flows through h → delta → loss."),
    ("Cosine similarity metric in KV-cache channel", "Channel 20 finds max cosine similarity between the current token's PPMI embedding and tokens in a 128-step window. What pattern does this detect? What's the steerer supposed to do with it?",
     "It detects topical cohesion. If 'Paris' appears 5 times in a paragraph, each occurrence has high cosine similarity with the others. This signals 'we are still discussing France.' The steerer uses this to modulate: during cohesive sections, trust local n-grams; during topic shifts, switch to global topic prior. This is our key-query-free memory retrieval channel."),
    ("Exponential decay in FastNgramFeatures", "Unigram counts multiply by 0.999 every 10 steps. What effective memory length does this create? Why not a simple sliding window?",
     "Half-life ≈ 693 steps ≈ 83000 tokens. A sliding window of 128 discards everything beyond it. Exponential decay gives smooth recency: nothing is fully forgotten, but old counts matter less. This captures both immediate context (recent tokens, high weight) AND long-range patterns (old tokens, low but non-zero weight) without storing history."),
    ("Orthogonal penalty purpose", "Why penalize mean((S@S^T - I)^2) for steer vectors S? What happens at 0.001 weight vs 0.1 weight?",
     "Drives steer vectors to be uncorrelated — each channel encodes independent information. At 0.001: weak penalty, channels can slightly overlap but won't collapse. At 0.1: strong penalty forces strict orthogonality, which can prevent useful correlations (e.g., punct_density naturally correlates with topic in code)."),
    ("Gamma parameter meaning in injection", "gammas start at 0.05. At inference, a gamma of 0.5 means what physically for the residual stream? What would gamma=2.0 imply?",
     "gamma=0.5: the steerer contributes ~50% as much energy as the original hidden state (after RMS normalization). The model's own computation still dominates. gamma=2.0: the steerer overwhelms the hidden state — the model's computation is nearly irrelevant. This happens when the cartridge overfits."),
    ("RMS normalization in _steer_layer", "Why scale the offset by h_rms / o_rms? What would happen if we removed this normalization?",
     "Preserves relative contribution: the steered delta matches the hidden state's energy scale. Without it: at layer 0, small h gets a huge relative delta (steerer dominates). At layer 10, large h overrides the steerer (steerer has no effect). RMS normalization makes injection CONSISTENT across layers with different activation scales."),
    ("NF4 vs uniform quantization", "NF4 divides the normal distribution into equiprobable regions. Why not uniform bins? What's the benefit for transformer weights?",
     "Neural weights are ~normal distributed (μ≈0, σ≈0.1-0.5). Uniform bins waste levels on tails (±3σ) where few weights exist while center gets too-few levels. NF4 gives each of 16 levels ~6.25% of weights — fine resolution at center, coarse at tails. For transformer FFN weights (normally distributed), this minimizes reconstruction error per bit."),
    ("Maxwell M40 vs RTX 3080 compute", "M40 (Maxwell, 5.2) takes ~570s/epoch vs 3080 at ~55s. Why 10x? Break down the factors.",
     "1) CUDA cores: 3072 @ 1.11GHz vs 8704 @ 1.71GHz → ~4.6x fp32 throughput. 2) Memory bandwidth: 288 GB/s GDDR5 vs 760 GB/s GDDR6X → ~2.6x. Attention is bandwidth-bound. 3) Tensor cores: 3080 has them (2-4x matmul acceleration), M40 has none. 4) Combined: visible 10x. But M40 has 24GB vs 10GB — for 500M+ models, it's the only option."),
    ("plateau diagnosis for 37.2 eval_b", "124M 37ch model stalled at eval_b=37.2 for 30 epochs at LR=5e-6. What's the bottleneck and three fixes?",
     "Bottleneck: 124M params exhausted on 119M tokens (1:1 ratio). Fixes: 1) Add C4 data to increase effective tokens. 2) Selective channel freezing — freeze converged local/mid, force gradients through global. 3) Warm-start from V2 (eval_b=32) and fine-tune — saves 200 epochs."),
    ("OOM debugging: 20MB allocation failure on 10GB", "OOM on 20MB allocation with 10GB GPU free. Why? What do you check?",
     "Memory fragmentation — many small free blocks but none contiguous for 20MB. Check: torch.cuda.memory_summary(), nvidia-smi (other processes?), leftover zombie training processes. Fix: empty_cache(), restart Python process, or PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."),
    ("bitsandbytes blocksize parameter", "Linear4bit uses blocksize=64. What does this control? Tradeoffs of 32 vs 256?",
     "NF4 quantizes in blocks of N weights, each block gets its own absmax scalar. blocksize=64: storage = 64×4 + 32 = 288 bits → 4.5 bits/weight. blocksize=32: 4.75 bits/weight, better accuracy, more overhead. blocksize=256: 4.125 bits/weight, worse accuracy (more weight variation per block), less overhead. 64 balances accuracy and storage for transformer FFN where weights have ~diagonal correlation within layers."),
    ("Gradient accumulation mechanics", "accumulate=4 splits a batch=32 into four batch=8 micro-batches. The gradients ADD, then optimizer.step() runs once. How does this differ from batch=32 directly?",
     "Mathematically identical for loss (sum of per-sample losses). Different in: 1) BatchNorm statistics (each micro-batch sees different mean/var), 2) Dropout masks (different per micro-batch), 3) Memory (8 vs 32 fits in GPU). Our models use LayerNorm (not BatchNorm) and no dropout, so accumulate is equivalent to larger batch for the same total token count."),
    ("Saving ZeroQ checkpoints", "zeroq_save_checkpoint() calls gather() before torch.save() and release() after. What state is saved without gather()? Why can't it resume?",
     "Without gather(): state_dict has 0-sized tensors for FFN layers (weights are in 4-bit shards). Loading these into a fresh model fails with size mismatch. gather() materializes FP32 weights into state_dict temporarily. release() frees them. A saved checkpoint with gather() can reload directly with model.load_state_dict()."),
    ("StreamingDataset: worker isolation", "Why does each DataLoader worker need its own RNG state, tokenizer, and feature computer?",
     "Workers are separate processes. Shared RNG → all workers produce identical batches (waste). Shared tokenizer → internal cache causes race conditions. Shared feature computer → worker A writes a token, worker B reads it thinking it's from its own context (corruption). Each worker needs isolated state to produce independent, valid sample streams."),
]

for title, q, a in THEORY:
    add(4, "theory", f"# {title}\n\n{q}\n\n", a,
        "import sys;c=sys.stdin.read();assert len(c)>100;print('PASS')",
        "deep_theory", title)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    for c in C: f.write(json.dumps(c) + "\n")
tiers = {}
for c in C: tiers[c["tier"]] = tiers.get(c["tier"], 0) + 1
print(f"Total: {len(C)} | Tiers: {tiers} | Saved to {OUT}")
