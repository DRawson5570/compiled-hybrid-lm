#!/usr/bin/env python3
import json, ast, glob, os
from pathlib import Path

REPO = Path("/home/drawson/deepseek_experiments")
OUT = Path("/home/drawson/code_harness/challenges/challenges_full.jsonl")
C = []

def add(tier, cat, prompt, expected, test, src, title=""):
    C.append({"id": f"{cat}_{len(C):04d}", "tier": tier, "category": cat,
              "prompt": prompt, "expected": expected,
              "test_code": test, "source_file": src, "title": title})

# 1. CODE: Every function in repo
for scan in ["hybrid", "experiments", "tests"]:
    for py in sorted(REPO.rglob(f"{scan}/**/*.py")):
        rid = str(py.relative_to(REPO))
        if "archive" in rid or "__pycache__" in rid: continue
        try: src = py.read_text()
        except: continue
        try: tree = ast.parse(src)
        except: continue
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
            if len(n.body) < 3: continue
            lines = src.split("\n")
            bs, be = n.body[0].lineno - 1, n.body[-1].end_lineno
            if be - bs < 2: continue
            preamble = lines[max(0, n.lineno - 4):bs]
            prompt = "\n".join(preamble) + "\n"
            expected = "\n".join(lines[bs:be]) + "\n"
            test = f'import ast,sys\ncode=sys.stdin.read()\ntry:\n t=ast.parse(code)\n f=[x.name for x in ast.walk(t) if isinstance(x,ast.FunctionDef)]\n assert "{n.name}" in f,f"Missing {n.name}, got {{f}}"\n print("PASS")\nexcept SyntaxError as e:print(f"FAIL:syntax {{e}}")\nexcept AssertionError as e:print(f"FAIL: {{e}}")\n'
            add(1, "code", prompt, expected, test, rid)

# 2. CODE: Full file chunks
for pattern in ["hybrid/*.py", "experiments/*.py"]:
    for fp in sorted(REPO.glob(pattern)):
        if fp.stat().st_size < 500 or fp.stat().st_size > 15000: continue
        src = fp.read_text()
        lines = src.split("\n"); n = len(lines)
        for chunk_start in range(0, n, n//3):
            prompt = "\n".join(lines[:chunk_start]) + "\n" if chunk_start > 0 else ""
            if len(prompt) < 20: continue
            expected = "\n".join(lines[chunk_start:chunk_start+n//6]) + "\n"
            if len(expected) < 10: continue
            test = "import ast,sys\ntry:\n ast.parse(sys.stdin.read());print('PASS')\nexcept SyntaxError as e:print(f'FAIL:{e}')\n"
            add(2, "file_chunk", prompt, expected, test, str(fp.relative_to(REPO)))

# 3. SHELL scripts
for sh in sorted(REPO.glob("*.sh")):
    src = sh.read_text(); lines = src.split("\n")
    for i in range(0, len(lines), max(len(lines)//3, 5)):
        prompt = "\n".join(lines[:i]) + "\n" if i > 0 else ""
        expected = "\n".join(lines[i:i+15]) + "\n"
        if len(prompt) < 10 or len(expected) < 10: continue
        test = "import sys;c=sys.stdin.read();assert len(c)>10;print('PASS')\n"
        add(3, "shell", prompt, expected, test, sh.name)

# 4. THEORY: compact Q&A bank  
THEORY_Q = [
    ("AdamW: why weight_decay not L2", "Explain why AdamW decouples weight decay from gradient-based updates, and why this matters for our 21-channel steerer with channels at vastly different gradient scales.",
     "AdamW applies weight decay directly to weights after the Adam update: w -= lr * (adam_update + wd * w). This means weight decay is uniform regardless of per-parameter gradient scale. In our 21-channel steerer, local n-gram channels have gradients ~100x larger than global topic channels. With L2-in-Adam, the large-gradient channels would experience less effective decay (decay competes with large gradients), letting them overfit while starving small-gradient channels. AdamW ensures ALL channels regularize equally."),
    ("Autograd: hook gradient flow explained", "A frozen DeepCausalLM has forward hooks registered by FeatureConditionedAdapterSteerer. Model params have requires_grad=False. Does loss.backward() compute gradients for the steerer? Why?",
     "Yes. The model's forward pass creates tensors (activations) that participate in the autograd graph even though the parameters producing them don't require grad. The steerer's _steer_layer takes these tensors as input and performs its own operations (down, feature, up). Since the steerer's parameters have requires_grad=True and the input tensors are part of the graph (not detached), the backward pass flows gradients through the steerer's computations to its parameters."),
    ("Feature: n-gram exponential decay", "FastNgramFeatures multiplies unigram counts by 0.999 every 10 steps. What effective context window does this create? Why not just use a sliding window?",
     "After 693 steps (ln(0.5)/ln(0.999) ≈ 693), counts halve. At ~120 tokens/step, that's ~83000 tokens of effective memory. A sliding window of 128 tokens would discard ALL statistics from earlier context. The exponential decay provides a smooth recency weighting: recent tokens count more, old tokens count less, but nothing is fully forgotten. This captures both immediate context (for local features) and long-range patterns (for global features) without storing the full history."),
    ("Superposition: orthogonal_penalty design", "orthogonal_penalty() computes mean((steer_vectors @ steer_vectors^T - I)^2). What does this optimize? What happens without it?",
     "Optimizes for uncorrelated steer vectors. Without it: if steer_local[0] and steer_local[1] point in similar directions, both encode overlapping information — the 21 channels collapse to fewer effective dimensions. The penalty drives vectors apart, maximizing the information content of each channel. At 0.001 weight in the loss, it's weak enough that useful correlations survive but strong enough to prevent collapse."),
    ("Cartridge: gamma and injection strength", "gammas start at 0.05. At inference after training, a gamma might be 0.5. What does gamma=0.5 mean physically for the residual stream?",
     "The steerer produces a delta via: h + gamma * alpha * beta * normalized_offset. gamma=0.5 means the steered delta contributes ~50% as much as the original hidden state in terms of RMS-normalized magnitude. The alpha (global, ~1.0) and beta (per-group, ~1.0) further modulate. gamma values > 1.0 would mean the steerer DOMINATES the hidden state — the model's own computation is nearly irrelevant. gamma < 0.01 means the cartridge is dormant."),
    ("ZeroQ: 4-bit NF4 vs FP16 tradeoff", "NF4 stores weights in 4 bits vs FP16's 16 bits. What's lost? When is this loss unacceptable?",
     "NF4 loses 3/4 of weight precision — each value can only be one of 16 levels. For the FFN layers of a 2B model, the reconstruction error is ~0.5-1% per weight. Cumulatively across 24 layers, this adds ~5-10% PPL increase. Unacceptable when: 1) training from scratch (the noise prevents convergence), 2) very small models (<100M, the quantization noise is proportionally larger), 3) low-precision tasks (arithmetic, exact lookups). Acceptable for: inference on large models, fine-tuning with frozen backbone."),
    ("Training: plateau at 37.2 diagnosis", "The 124M 37ch model stalled at eval_b=37.2 for 30 epochs at LR=5e-6. Identify the most likely bottleneck and three actionable fixes.",
     "Bottleneck: model capacity exhausted — 124M params can only represent PPL ~37 on 119M WikiText tokens given the compiled prior headroom. Fixes: 1) Add data: C4 or more WikiText to increase effective training tokens beyond 1:1 ratio. 2) Selective channel freezing: freeze converged local/mid channels, force gradients through global channels. 3) Warm-start: load a stronger base model (V2 at eval_b=32 already exists) and fine-tune — saves 200 epochs of convergence."),
    ("Streaming: IterableDataset worker isolation", "Why does each DataLoader worker need its own RNG, tokenizer, and feature computer? What happens if they share?",
     "Workers are separate processes. If they share an RNG seeded identically, all workers produce the same sequence — wasting batch slots. If they share a tokenizer, the tokenizer's internal cache causes race conditions (token IDs change between workers). If they share a feature computer, n-gram state gets corrupted (worker A writes token X, worker B reads it thinking it's from a different context). Isolation ensures each worker builds its own statistically independent sample stream."),
    ("GPU: Maxwell M40 training speed vs 3080", "Why is a 124M training epoch 55s on RTX 3080 but ~570s on M40? Where does the 10x come from?",
     "3080 (Ampere): 8704 CUDA cores @ 1.71 GHz, tensor cores for fp16 matmul, GDDR6X @ 760 GB/s. M40 (Maxwell): 3072 CUDA cores @ 1.11 GHz, NO tensor cores, GDDR5 @ 288 GB/s. The 10x: 1) Compute throughput: 29.8 vs 6.8 TFLOPS fp32 (~4.4x). 2) Memory bandwidth: 760 vs 288 GB/s (~2.6x) — transformer layers are bandwidth-bound. 3) Tensor cores accelerate attention matmuls 2-4x on 3080. Combined: ~10x. But M40 has 24GB vs 10GB on 3080 — for large models, M40 is the only option."),
    ("CUDA: OOM debugging strategy", "You get 'CUDA out of memory. Tried to allocate 20 MiB.' Only 20MB was requested but the GPU has 10GB. Why the OOM? What do you check?",
     "The 20MB request failed because GPU memory is FRAGMENTED — many small free blocks exist but none contiguous for the allocation. Check: 1) torch.cuda.memory_summary() for fragmentation details. 2) Is another process using the GPU (nvidia-smi)? 3) Did a previous run leave unreleased memory (process zombie)? 4) Is PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set? Fixes: empty_cache(), restart Python, or set expandable_segments. In our case, it's usually process 4279 (Xorg) + leftover training processes."),
    ("NCCL: all_reduce vs all_gather", "ZeroQ uses all_gather for weight shard assembly. Why all_gather instead of all_reduce? What's the difference in memory?",
     "all_gather concatenates shards from all ranks: each rank sends its shard (size S) and receives all shards (size S*N). Memory: input S, output S*N. all_reduce sums tensors across ranks and distributes: input = output size. For weight assembly, we need the FULL weight matrix (concatenation), not the sum — we're gathering pieces, not reducing gradients. Memory: all_gather needs S*N at output, which for a 2B model with 4 ranks means 2GB of communication. With only 1 GPU, both do nothing."),
    ("Bitsandbytes: what blocksize means", "Linear4bit uses blocksize=64. What does this parameter control? What's the tradeoff?",
     "NF4 quantization works on BLOCKS of 64 weights. Each block gets its own absmax scaling factor (stored in fp32). Blocksize=64: 64 weights × 4 bits = 256 bits + 32-bit scale = 4.5 bits/weight effective. Blocksize=256: 256×4 + 32 = 4.125 bits/weight (more efficient) but coarser quantization (more weight variation within a block). Smaller blocks: better accuracy, larger overhead. 64 is the PyTorch default and works well for transformer FFN layers where weights have block-diagonal correlation structures."),
    ("HuggingFace: trust_remote_code needed for Qwen3.5", "Qwen3.5 requires trust_remote_code=True to load. What custom code does it include? What security risk does this pose?",
     "Qwen3.5 includes custom modeling code: chunk_gated_delta_rule attention mechanism and custom rotary embeddings. This code is loaded and EXECUTED from the HF cache during from_pretrained(). Security risk: a malicious model could include arbitrary Python in its modeling file. Mitigation: 1) Only load from trusted sources (Qwen is Alibaba, verified). 2) Inspect the modeling file before loading. 3) Use HF's safetensors format (serialized weights only, no code). Our setup uses it because it's the official Qwen implementation."),
    ("BPE: encode vs encode_plus vs tokenizer()", "AutoTokenizer has encode(), encode_plus(), and __call__(). When to use each in our codebase?",
     "encode() returns token IDs list. encode_plus() returns a dict with input_ids, attention_mask, optionally offset_mapping. __call__() is batch-aware and handles padding/truncation. We use encode() for simple prompt encoding (chatbot.py). encode_plus() with return_offsets_mapping=True for NER feature alignment (ner_features.py). __call__() is not used because our sequences don't need batching or padding."),
    ("Training: best vs current eval tracking", "train_steerer_v4.py tracks both best_eval_b and current eval_b. Why save on best_eval_b instead of the last epoch?",
     "The model's eval PPL can BOUNCE: epoch 150 might be 47.1, epoch 151 might be 48.2 (worse), epoch 152 might be 46.9 (best). At epoch 200, the model could be at 48.5 (worse than best). Saving on best_eval_b ensures we keep the OPTIMAL checkpoint. Without this, a bad batch at the last epoch could overwrite a good model. The cost: best checkpoint is stale (epoch 152, not 200). We resume from best checkpoint for continued training."),
    ("Cartridge: composition mode additive vs mean vs chain", "The rack supports three composition modes. When would you use each for a multi-cartridge chatbot (chat + knowledge + code)?",
     "Additive: all cartridges act independently on the hidden state. Use when cartridges encode orthogonal skills (chat style vs code syntax vs factual tone). Risk: competing modifications can cancel or amplify noise. Mean: averages deltas from all active cartridges, reducing noise but also reducing each cartridge's effectiveness. Use when many cartridges are active and additive would saturate. Chain: cartridges apply sequentially, each seeing the previous one's output. Use when cartridges have a natural pipeline: format correction → content steering → style polishing. Our production chatbot uses additive by default."),
]

for title, q, a in THEORY_Q:
    add(4, "theory", f"# {title}\n\n{q}\n\n", a,
        "import sys;c=sys.stdin.read();assert len(c)>100;print('PASS')\n",
        "CMI_CODEBASE_DEEPDIVE.md", title)

# Save
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    for c in C: f.write(json.dumps(c) + "\n")
tiers = {}; [tiers.update({c["tier"]: tiers.get(c["tier"],0)+1}) for c in C]
print(f"Total: {len(C)} challenges | Tiers: {tiers} | Saved to {OUT}")
