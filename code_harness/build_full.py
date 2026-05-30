#!/usr/bin/env python3
"""Build 500+ challenge dataset from every file in the repo."""
import json, ast, os, glob as gb
from pathlib import Path

REPO = Path("/home/drawson/deepseek_experiments")
OUT = Path("/home/drawson/code_harness/challenges/challenges_full.jsonl")
COUNTER = [0]

def add(challenges, tier, cat, prompt, expected, test, src, title=""):
    COUNTER[0] += 1
    challenges.append({
        "id": f"{cat}_{COUNTER[0]:04d}",
        "tier": tier, "category": cat,
        "prompt": prompt, "expected": expected,
        "test_code": test, "source_file": src,
        "title": title or src,
    })

# --- Code extraction: every .py file ---
def extract_all_code():
    ch = []
    for scan_dir in CODE_DIRS:
        for py_file in sorted(REPO.rglob(f"{scan_dir}/**/*.py")):
            rid = py_file.relative_to(REPO)
            if "archive" in str(rid) or "__pycache__" in str(rid): continue
            try: source = py_file.read_text()
            except: continue
            try: tree = ast.parse(source)
            except: continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
                if len(node.body) < 3: continue
                lines = source.split("\n")
                body_start = node.body[0].lineno - 1
                body_end = node.body[-1].end_lineno
                if body_end - body_start < 2: continue

                sig = lines[node.lineno - 1:body_start]
                prompt = "\n".join(lines[max(0,node.lineno-4):node.lineno-1] + sig) + "\n"
                expected = "\n".join(lines[body_start:body_end]) + "\n"
                func = node.name
                test = f'''import ast,sys
code=sys.stdin.read()
try:
 t=ast.parse(code)
 f=[n.name for n in ast.walk(t) if isinstance(n,ast.FunctionDef)]
 assert "{func}" in f,f"Missing {func}, got {{f}}"
 print("PASS")
except SyntaxError as e:print(f"FAIL:syntax {{e}}")
except AssertionError as e:print(f"FAIL: {{e}}")
'''
                add(ch, 1, "code", prompt, expected, test, str(rid), f"Implement {func}()")

    # Full file challenges
    for path_str in gb.glob(str(REPO / "hybrid/*.py")) + gb.glob(str(REPO / "experiments/*.py")):
        fpath = Path(path_str)
        if fpath.stat().st_size < 500 or fpath.stat().st_size > 20000: continue
        src = fpath.read_text()
        lines = src.split("\n")
        mid = max(len(lines)//3, 5)
        prompt = "\n".join(lines[:mid]) + "\n"
        expected = "\n".join(lines[mid:mid+50]) + "\n"
        test = "import ast,sys\ntry:\n ast.parse(sys.stdin.read());print('PASS')\nexcept SyntaxError as e:print(f'FAIL:{e}')"
        add(ch, 2, "code_full", prompt, expected, test, str(fpath.relative_to(REPO)))

    # Shell scripts
    for sh in sorted(REPO.glob("*.sh")):
        src = sh.read_text()
        lines = src.split("\n")
        mid = max(len(lines)//2, 5)
        prompt = "\n".join(lines[:mid]) + "\n"
        expected = "\n".join(lines[mid:min(mid+30,len(lines))]) + "\n"
        test = "import sys;c=sys.stdin.read();assert '#!/bin/bash' in c or 'set -' in c or len(c)>20;print('PASS')"
        add(ch, 3, "shell", prompt, expected, test, sh.name)

    return ch

# --- Massive theory bank ---
THEORY = [
    # PyTorch deep internals
    ("PyTorch: nn.Module._call_impl hook lifecycle", "When forward() is called on an nn.Module, _call_impl runs pre-hooks, forward, then post-hooks. In our rack, hooks are registered via register_forward_hook on encoder layers. Are these pre-hooks or post-hooks? When in the lifecycle does the steerer modify hidden states?", "These are forward HOOKS (not pre-hooks). They fire AFTER the module's forward completes. In _call_impl: pre_hooks run first, then forward(), then forward hooks. Our steerer hook receives the layer's OUTPUT (post-attention + post-FFN + residual). The hook modifies this output and returns it. The returned value REPLACES the layer's output for downstream layers."),
    ("Autograd: retain_graph and multiple backward passes", "If loss.backward() is called twice without retain_graph=True, what error occurs? When would you need retain_graph in our trainer?", "Error: 'Trying to backward through the graph a second time, but saved intermediate results have already been freed.' The autograd graph is freed after backward() by default. retain_graph=True is needed if you: 1) need second-order gradients, 2) have multiple losses sharing the same computation, or 3) debugging by inspecting gradients. Our trainer uses a single loss.backward() per step so doesn't need it."),
    ("PyTorch: Parameter vs Buffer distinction", "Why does register_buffer exist? The _semantic_proj was originally a buffer and now is a trainable Parameter. What's the practical difference?", "Parameters: included in model.parameters(), returned by state_dict(), updated by optimizers. Buffers: included in model.buffers(), returned by state_dict(), NOT updated by optimizers. _semantic_proj was a buffer (requires_grad=False) because we thought the random projection doesn't need training. It was wrong — we changed it to a trainable Parameter (semantic_encoder) so gradients flow through it."),
    ("Transformer: scaled dot-product attention formula", "Write the full formula for scaled dot-product attention and explain each term. Why divide by sqrt(d_k)?", "Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V. Q,K,V: (B, H, T, d_k). Q @ K^T: (B, H, T, T) — pairwise similarity between all positions. / sqrt(d_k): scale to prevent dot products from growing with d_k. If d_k=64, dot products can be ±64, making softmax saturate to one-hot. sqrt(64)=8 normalizes to ~±8, keeping softmax in a useful range. @ V: weight values by attention scores."),
    ("Transformer: LayerNorm vs BatchNorm", "Why does Transformer use LayerNorm? What would happen with BatchNorm at test time when batch_size=1?", "LayerNorm normalizes across the FEATURE dimension per sample: mean/std of d_model values at each position. BatchNorm normalizes across the BATCH dimension per feature. With batch_size=1 at inference, BatchNorm's running mean/variance would dominate, and the single sample would get normalized to approximately the training distribution mean — losing per-sample variation. LayerNorm handles any batch size because it's per-sample. Also: sequences have variable length; BatchNorm would require padding handling. LayerNorm naturally handles this."),
    ("AdamW: epsilon parameter", "AdamW has eps=1e-8. What does epsilon prevent? Why not eps=0?", "eps prevents division by zero in the update: m_hat / (sqrt(v_hat) + eps). If v_hat (running variance) approaches zero (flat gradients), sqrt(v_hat) ≈ 0 and the update would explode. eps=1e-8 is the default; values like 1e-3 are used in some setups. Too large eps dampens adaptivity."),
    ("Gradient: torch.no_grad() vs torch.inference_mode()", "Our generate method uses torch.no_grad(). What additional optimization does torch.inference_mode() provide? Why don't we use it?", "inference_mode() also disables autograd tracking like no_grad(), PLUS disables version counter bumps on tensors. Version counters are used to detect in-place modifications that would invalidate saved tensors. In inference_mode, these checks are skipped, improving performance ~5-10%. We don't use it because: 1) it was introduced in PyTorch 1.9, not available on older m40_env PyTorch; 2) some operations that check version counters (like tensor.view()) can fail in inference_mode."),
    ("CUDA: cudaMalloc vs cudaMallocAsync", "PyTorch uses cudaMalloc by default. When would cudaMallocAsync (CUDA 11.2+) help our training on the 3080?", "cudaMallocAsync allows memory allocation to overlap with GPU computation. In our training loop: step N's backward runs while step N+1's forward allocates memory for activations. Without async: the allocation blocks the compute. With async: they overlap. The benefit is small (~5%) for our workloads because allocation time is dwarfed by compute time. More useful for inference with dynamic shapes."),
    ("DataLoader: pin_memory internals", "pin_memory=True allocates tensors in page-locked memory. Why is page-locked memory faster for CPU->GPU transfers? What's the downside?", "Page-locked memory cannot be swapped to disk. The DMA engine can directly access it without the CPU copying from pageable memory first. This eliminates a memcpy: pageable → pinned (CPU) → GPU vs pinned → GPU directly. Downside: too much pinned memory starves the OS of pageable pages, causing OOM at the system level. Our DataLoader pins small batches (8*128*2 bytes ~ 2KB), negligible risk."),
    ("Loss: perplexity interpretation", "Eval PPL of 46 means the model is as 'surprised' as if choosing uniformly from 46 options. Is this good for a 124M model?", "GPT-2 Small (124M) achieves PPL ~35 on WikiText-103 after full training. Our best from-scratch is 37.2 (with semantic channels). Warm-started V4 is 32. So 46 is reasonable for from-scratch with limited tokens. Human-level PPL on WikiText is ~10-12. State-of-the-art 7B+ models achieve PPL <10. At 124M with 119M tokens (1:1 ratio), we're data-starved, not capacity-limited."),

    # Our architecture specifics
    ("Steerer: why three separate MLPs", "local_mlp, mid_mlp, global_mlp are separate. Why not one 21-channel MLP? What would happen?", "One MLP(21→42→21) would learn cross-group correlations: a local n-gram feature could influence global steering, and vice versa. This sounds useful but HURTS at overfit scale: the MLP has 21*42+42*21 = 1,764 weights for 119M tokens — it easily memorizes spurious correlations (e.g., 'when entropy is high AND topic is science, boost token X'). Separate MLPs force each group to produce deltas independently, reducing parameter count per group and preventing cross-group overfitting. The routing (local→early, global→deep) would be meaningless with cross-channel mixing."),
    ("Compiled features: ppmi_embeddings initialization", "ppmi_embeddings is initialized as torch.randn(V, 256) * 0.01. Why 256 dimensions? Why 0.01 scale?", "256 is a design choice balancing information and overfitting. V=50257, so 256 is ~5% of vocab size — enough to capture semantic similarity without being a full embedding matrix. 0.01 scale ensures initial PPMI similarities are small (dot product ~0.01^2 * 256 = tiny), so early training relies on other channels. As training progresses, the steer vectors learn to amplify useful PPMI patterns."),
    ("GPU: memory fragmentation and empty_cache()", "Our trainer calls torch.cuda.empty_cache() after eval. What does this actually do? When is it useless?", "empty_cache() releases CUDA memory from PyTorch's caching allocator back to the OS. PyTorch caches freed memory for reuse — this is normally good (avoids re-allocation). After eval (which allocates different-shaped tensors than training), the cached memory pool is fragmented (many small free blocks). empty_cache() releases all cached blocks, forcing fresh allocations for training. Useless when: 1) memory usage is stable (no fragmentation), 2) the next allocation fits in existing cached blocks. Overuse hurts performance (re-allocation is slow)."),
    ("Training: eval_tokens parameter", "eval_tokens defaults to 8192. The validation set has 247K tokens. Why evaluate on only 3.3%? What's the tradeoff?", "247K tokens takes ~3 seconds per eval pass. At 500 steps per epoch, this adds 3s/55s = 5% overhead. eval_tokens=8192 reduces this to ~0.1s (0.2% overhead). The tradeoff: stochastic eval (higher variance per epoch) vs faster epochs. With 8192 tokens, the standard error of PPL is ~1/sqrt(8192/avg_token_count) ≈ 1/sqrt(8192/60) ≈ 0.09. So eval_b=46.0 ± 0.1 — accurate enough for tracking improvements. Full eval at epoch 200 gives the precise number."),
    ("Feature: KV-cache channel motivation", "Channel 20 (KV-cache cosine similarity) computes max cosine similarity between current token's PPMI embedding and all previous tokens in a 128-token window. What pattern does this detect?", "This detects self-reference and topical cohesion. If the model is discussing 'France' and uses 'Paris' 5 times, the cosine similarity between 'Paris' PPMI embeddings will be high at each occurrence. This signals 'we are still on this topic.' Conversely, a topic shift produces low cosine similarity. The steerer uses this to modulate prediction: during cohesive passages, rely on local n-grams; during topic shifts, switch to global topic prior. This is our version of a 'memory retrieval' channel."),

    # System and infra
    ("Screen: session management", "Why use screen instead of nohup for long training runs? When does nohup fail where screen succeeds?", "screen provides a persistent terminal session that survives SSH disconnection AND allows reattachment. nohup only ignores SIGHUP but doesn't provide reattachment — you can't check the process's stdout interactively after launch. screen also protects against terminal hangup signals that nohup can't (e.g., closing the terminal emulator). Our pattern: launch in screen, detach, and reattach anytime to monitor. Screen also runs in its own process group, isolating it from parent shell termination."),
]
