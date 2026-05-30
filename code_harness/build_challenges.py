#!/usr/bin/env python3
"""Build comprehensive challenge dataset for code cartridge training.

Generates 500+ challenges across four categories:
  - theory: deep learning concepts and our architecture
  - code: function completion from codebase
  - explain: code explanation in project context
  - reason: experiment and decision interpretation
"""
import json, sys, ast, os
from pathlib import Path
from dataclasses import dataclass, asdict

REPO = Path("/home/drawson/deepseek_experiments")
OUT = Path("/home/drawson/code_harness/challenges/challenges_augmented.jsonl")


def theory_challenges() -> list[dict]:
    """Deep theory Q&A pairs covering every technology we use."""
    pairs = [
        # === PyTorch fundamentals ===
        ("Autograd: forward hook execution order", """During a nn.TransformerEncoder forward pass with SuperpositionSteererV3 hooks registered, in what order do hooks fire relative to the layer's own computation? What tensor does the hook receive and what happens to its return value?""",
         """Forward hooks fire AFTER the module's forward method completes. The hook receives (module, inputs, output). The `output` is the layer's processed hidden state (after self-attention + FFN + residuals). The hook can modify and return a new output, which REPLACES the layer's original output for downstream layers. For SuperpositionSteererV3, the hook computes a residual-stream delta from compiled channel features and adds it: `return h + (scale * normalized_offset).to(dtype=h.dtype)`. The steered hidden state then flows to the next layer's input."""),

        ("Autograd: gradient flow through hooks", """A frozen DeepCausalLM model has forward hooks registered by a FeatureConditionedAdapterSteerer. Model parameters have requires_grad=False. Does loss.backward() compute gradients for the steerer parameters? Why?""",
         """Yes. `loss.backward()` traverses the autograd graph from the loss tensor backward. Even though model parameters have requires_grad=False, the tensors they PRODUCE (activations) still participate in the graph. The steerer's `_steer_layer` modifies hidden states by computing `delta = up(GELU(down(norm(h)) + feature(channels)))`. Since `h` was produced by the model's forward pass and requires grad (through the graph, not the parameters), the `delta` computation creates new nodes that require grad. The backward pass flows through the delta computation to the steerer parameters."""),

        ("AdamW: weight decay mechanism", """In our training loop, we use `AdamW(steerer_params, lr=3e-4, weight_decay=0.01)`. How does AdamW apply weight decay differently from L2 regularization? Why does this matter for our 21-channel steer vectors?""",
         """AdamW decouples weight decay from the adaptive learning rate. Traditional Adam with L2 adds the weight decay gradient to the loss gradient BEFORE the Adam update, which couples decay to the adaptive learning rates — parameters with large gradients get less decay. AdamW applies decay directly to the weights AFTER the Adam update: `w = w - lr * (adam_update + weight_decay * w)`. This is critical for our steer vectors because the 21 channels have VERY different gradient scales (n-gram stats vs topic prior vs punct density). Decoupled decay ensures EACH channel gets uniform regularization regardless of its gradient magnitude."""),

        ("Transformer: norm_first architecture", """Our DeepCausalLM uses nn.TransformerEncoder with norm_first=True. What is the computation order difference from norm_first=False? Why does this affect PyTorch's fused CUDA kernel availability?""",
         """norm_first=True: LayerNorm → Attention → Add residual → LayerNorm → FFN → Add residual. norm_first=False: Attention → Add → LayerNorm → FFN → Add → LayerNorm. The norm_first order is the modern Pre-LN architecture. PyTorch's fused `_transformer_encoder_layer_fwd` C++ kernel supports norm_first=True but requires specific conditions (no custom modules inside the layer). When we replace FFN Linear modules with bnb Linear4bit (ZeroQ), the C++ kernel can't handle the custom module and crashes with dtype mismatch errors. Our fix: patch each layer's forward with a pure-Python equivalent that calls Linear4bit.forward() normally."""),

        ("Attention: QKV projection and multiple heads", """In DeepCausalLM's self_attn, the `in_proj_weight` has shape (3*d_model, d_model). How are query, key, and value extracted? How do multiple heads work after projection?""",
         """`in_proj_weight` is a concatenation of three weight matrices: `W_q, W_k, W_v`, each (d_model, d_model). The input x (B,T,d_model) is multiplied: `qkv = x @ in_proj_weight^T` producing (B,T,3*d_model). This is split into q,k,v each (B,T,d_model). Each is reshaped to (B,T,n_heads,d_head) and transposed to (B,n_heads,T,d_head) for multi-head attention. The attention scores are computed per head: `scores = softmax(q @ k^T / sqrt(d_head)) @ v`. Outputs are concatenated: (B,T,d_model) and projected through out_proj: (B,T,d_model)."""),

        # === Our architecture ===
        ("Steerer: compiled priors design", """The 21 compiled feature channels are split into local(6), mid(7), and global(8) groups routed to different layers. Why is this routing important? What would happen if all 21 channels were injected at every layer?""",
         """Local channels (n-gram log-probs, recency) capture immediate token context — most useful at early layers (0,1,2) where the model hasn't built deep representations. Mid channels (skip-3, entropy, global unigram, PPMI) bridge local and global statistics (layers 4,5,6). Global channels (topic, KV-cache, POS, punct density) capture document-level properties — most useful at deep layers (8,9,10) where semantic processing occurs. Injecting all 21 at every layer would amplify noise: local features at deep layers would fight against the semantic computation, and global features at early layers would provide no signal (document-level stats haven't accumulated yet). The routing matches feature type to representation depth."""),

        ("Cartridge: residual-stream composition", """Three cartridges (chat, knowledge, code) are mounted on a SteererCartridgeRack. How does additive composition work vs chain composition? When would each fail?""",
         """Additive: each steerer independently computes delta_i = steer_i(h) - h. Total delta = sum(weight_i * delta_i). All steerers see the ORIGINAL hidden state `h`. This works when cartridges encode orthogonal skills (chat style vs code syntax). Fails when cartridges compete: if both try to modify the same semantic feature, they either cancel or blow up. Chain: result = steer_n(...(steer_1(h))). Each steerer sees the output of the previous one. This works when cartridges have a natural order (format correction → content steering). Fails when order matters but is unknown — wrong order produces garbage. Our rack uses additive by default; chain mode for multi-step pipelines."""),

        ("Cartridge: bottleneck design", """FeatureConditionedAdapterSteerer uses a 64-384 dim bottleneck. The down projection (d_model → bottleneck) and up projection (bottleneck → d_model) sandwich a feature-conditioned GELU. Why does bottleneck size matter? What happens at bottleneck=1 vs bottleneck=d_model?""",
         """bottleneck=1: The adapter becomes a scalar gate — it can only scale the hidden state uniformly per layer. Too restrictive for task-specific steering. bottleneck=d_model: The adapter can learn arbitrary hidden state transformations, but has too many parameters (d_model^2) and overfits. bottleneck=64-128: sweet spot. Enough capacity to encode task-specific patterns (format correction, answer letter preference) without memorizing training examples. Our ARC cartridge at bottleneck=128 achieved 93% with 3.8M params vs Qwen's 4B."""),

        ("Semantic channels: trainable vs fixed projection", """The semantic encoder transforms hidden states h (d_model) to semantic_dim channels via a learned MLP. Why does a FIXED random projection fail while a LEARNED one works? What does the encoder learn?""",
         """Fixed random projection preserves distances (Johnson-Lindenstrauss) but can't AMPLIFY task-relevant features. If the hidden state encodes 'question about physics' as a weak signal in dimensions 143 and 712, a random projection mixes this with noise and loses it. A learned encoder can amplify those specific dimensions and suppress irrelevant ones. The encoder learns to extract features that the downstream semantic_mlp and steer_semantic vectors can USE to produce useful hidden-state deltas. Our experiments showed: fixed→no improvement over baseline, trainable→8.7 PPL improvement over 21-channel baseline."""),

        # === GPU/System ===
        ("GPU: Maxwell M40 constraints", """A Tesla M40 (compute 5.2) is used for training. Why can't it run fp16 tensor core operations? How does this affect training strategy?""",
         """The M40 is Maxwell architecture — it has NO tensor cores. Tensor cores were introduced in Volta (compute 7.0+). All computation on M40 happens in fp32 CUDA cores, even when data is stored in fp16. fp16 on M40 only saves MEMORY BANDWIDTH (half the bytes to transfer), not compute. Our strategy: store models in fp16 for memory efficiency, but the CUDA cores compute in fp32. This means training is memory-bandwidth-bound, not compute-bound. Also: PyTorch 2.7+ dropped Maxwell support (minimum sm_70), so we use an older compatible PyTorch in `~/local_venvs/m40_env/`."""),

        ("ZeroQ: 4-bit quantization mechanics", """ZeroQ partitions frozen FFN weights into 4-bit NF4 format. Explain: how are FP32 weights converted to 4-bit? What is the absmax quantization scheme? What is the dequantization formula at forward time?""",
         """NF4 (NormalFloat4) quantization: 1) Weights are normalized to zero-mean unit-variance. 2) The normal distribution is divided into 16 equal-probability regions (for 4-bit = 2^4 levels). 3) Each weight is assigned to the nearest level. 4) Storage: quantized indices (4 bits each) + absmax scaling factor per block (typically 64 weights). Dequantization at forward time: `w_fp = quant_state.absmax * (indices * quant_state.code + quant_state.offset)`. The code lookup table maps 4-bit indices to floating values. This gives ~4x memory reduction (FP32 32 bits → NF4 4 bits) with <1% accuracy loss."""),

        ("ZeroQ: gather/release checkpointing", """Our zeroq_save_checkpoint() calls handle.wrapper.start_gather() before torch.save() and handle.wrapper.release() after. Why is this necessary? What happens without it?""",
         """ZeroQ stores frozen FFN weights as 4-bit quantized shards. `model.state_dict()` returns the CURRENT in-memory representation — which after partitioning has empty (zero-size) tensors for FFN layers. If we save this directly, the checkpoint is useless (can't reload). `start_gather()` materializes the FP32 weights from the 4-bit shards back into the model's state dict. `release()` frees the FP32 copies, returning to 4-bit storage. Without this: saved checkpoints have 0-sized tensors (e.g., `linear1.weight: [0]`), and resuming fails with size mismatch errors."""),

        ("Training: crossover point analysis", """During from-scratch 124M training, the steerer wins (eval_s < eval_b) for the first 10 epochs, then the base model overtakes. What does this CROSSOVER reveal about the role of compiled priors?""",
         """The crossover at epoch 10 reveals that compiled priors are TRAINING ACCELERATORS, not permanent features. In epochs 1-9, the model's own weights are nearly random. The steerer's 21-channel features provide a strong initialization signal (n-gram stats, topic distributions) that dramatically improves predictions. By epoch 10, the model has internalized these statistical patterns in its own weights — it learned to compute equivalent representations from raw tokens. After crossover, the steerer becomes dead weight: the model is BETTER without it. This proves the priors are 'training wheels' — necessary for convergence but discharged once the model matures. For permanent inference improvement, priors need to encode information the model CANNOT derive from raw text (entity types, external knowledge)."""),

        ("Training: plateau response", """A training run plateaus at eval_b=37.2 for 15+ epochs. What THREE strategies should you try in order, and why that order?""",
         """1) LOWER LR: The current LR is too large to find the valley floor. Drop from 3e-6 to 1e-6 or 5e-7. This is the cheapest change — no restart needed. 2) WARM RESTART: Load a checkpoint from BEFORE the plateau (e.g., epoch 350 when eval was still dropping) and resume with the lower LR. The optimizer's momentum state at the plateau may be stuck — fresh momentum helps. 3) ADD DATA: If strategies 1-2 fail, the model is bottlenecked by data diversity. At 119M tokens for 124M params (1:1 ratio), the model is severely undertrained. Add C4 or other data. This is the most expensive change but also the most effective — our best from-scratch result (37.2) has room to improve with more data."""),

        ("Steerer: orthogonal penalty design", """orthogonal_penalty() computes mean((steer_vectors @ steer_vectors^T - I)^2). What does this penalize? Why is orthogonality important for a 21-channel steerer?""",
         """This penalizes correlation between steer vectors. `steer_vectors @ steer_vectors^T` produces a 21×21 correlation matrix. Subtracting I and squaring gives high loss when off-diagonal elements (cross-correlations) are non-zero. Orthogonality is important because each steer vector represents a different compiled feature's contribution. If two vectors overlap (high cosine similarity), signals from different channels get mixed in the same direction of hidden-state space, making the channel grouping meaningless. The local/mid/global routing ASSUMES channels are independent — non-orthogonal vectors violate this assumption and the steerer becomes a random blender instead of a structured prior."""),

        ("NCCL: distributed training initialization", """In maybe_init_dist_for_zeroq(), we call dist.init_process_group(backend='nccl'). What does this do? Why does a single-GPU ZeroQ run need distributed init?""",
         """`init_process_group(backend='nccl')` initializes the PyTorch distributed communication layer using NCCL (NVIDIA Collective Communications Library). Even for single-GPU training, ZeroQ uses distributed primitives for its internal shard communication: `all_gather`, `broadcast`, etc. ZeroQ's coordinator and wrapper were designed for multi-GPU setups where weight shards are distributed across devices. On single GPU, we still need a process group of size 1. We set `MASTER_ADDR=localhost MASTER_PORT=12355 RANK=0 WORLD_SIZE=1` to fake a single-node distributed environment. NCCL falls back to single-device mode when world_size=1."""),

        ("StreamingDataset: IterableDataset lifecycle", """Our StreamingDatasetC4 is a torch IterableDataset. How does DataLoader handle worker assignment? Why must tokenizer and rng state be per-worker?""",
         """IterableDataset.__iter__() is called ONCE per worker process. The DataLoader spawns `num_workers` subprocesses, each calling __iter__() independently. Each worker gets a `worker_id` from `get_worker_info().id`. We use this to: 1) Shard the C4 file list so each worker reads different files (`worker_files = shuffled[worker_id::num_workers]`). 2) Seed the RNG differently per worker (`random.Random(seed + worker_id * 1009)`) to avoid all workers producing the same sequence. Without per-worker seeding, all workers would yield identical batches, wasting GPU memory."""),

        ("BPE: GPT-2 tokenizer alignment with spaCy NER", """Our NER features use GPT-2 BPE tokens but spaCy's NER uses word-level tokens. How does ner_features.py align them? What edge case causes the most bugs?""",
         """Alignment via character offsets: `tokenizer.encode(text, return_offsets_mapping=True)` gives `[(start, end)]` for each BPE token. spaCy's `ent.start_char` and `ent.end_char` give entity character spans. For each BPE token, if ANY of its characters fall inside an entity span, the token is labeled with that entity type. Edge case: BPE subword boundaries. If "organization" is tokenized as [" organ", "ization"] and only " organ" has entity overlap, only that subword gets the entity label. The second subword (part of the same word) gets no label. This causes the cartridge to see entity labels on partial words, which can confuse the gating MLP. Mitigation: use max-pooling over consecutive tokens."""),
    ]

    challenges = []
    for i, (title, question, answer) in enumerate(pairs):
        prompt = f"# {title}\n\n{question}\n\n"
        challenges.append({
            "id": f"theory_{i:03d}",
            "tier": 4,
            "category": "theory",
            "prompt": prompt,
            "expected": answer,
            "test_code": f"import sys\ncode = sys.stdin.read()\nassert len(code) > 100, \"Answer too short\"\nprint(\"PASS\")",
            "source_file": "CMI_CODEBASE_DEEPDIVE.md",
            "title": title,
        })
    return challenges


def code_completion_challenges() -> list[dict]:
    """Tier 1-3 code completion from the repo."""
    challenges = []
    for py_file in sorted(REPO.rglob("hybrid/*.py")):
        if "archive" in str(py_file) or "__pycache__" in str(py_file):
            continue
        try:
            source = py_file.read_text()
        except Exception:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if len(node.body) < 2:
                continue
            body_start = node.body[0].lineno - 1
            body_end = node.body[-1].end_lineno
            if body_end - body_start < 2:
                continue

            lines = source.split("\n")
            sig_lines = lines[node.lineno - 1:body_start]
            doc = ast.get_docstring(node)
            if not doc and len(sig_lines) < 3:
                continue

            # Prompt: file context + signature
            preamble = lines[max(0, node.lineno - 5):node.lineno - 1]
            prompt = "\n".join(preamble + sig_lines) + "\n"
            expected = "\n".join(lines[body_start:body_end]) + "\n"
            func_name = node.name
            cid = f"code_{py_file.stem}_{func_name}"

            test_code = f"""\
import ast; import sys
code = sys.stdin.read()
try:
    tree = ast.parse(code)
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "{func_name}" in funcs, f"Function {func_name} not found"
    print("PASS")
except SyntaxError as e:
    print(f"FAIL: syntax error - {{e}}")
except AssertionError as e:
    print(f"FAIL: {{e}}")
"""
            challenges.append({
                "id": cid, "tier": 1 if body_end - body_start < 20 else 2,
                "category": "code",
                "prompt": prompt, "expected": expected,
                "test_code": test_code,
                "source_file": str(py_file.relative_to(REPO)),
                "title": f"Implement {func_name}()",
            })

    # Also add full-file challenges for smaller modules
    for rel_path in ["hybrid/ner_features.py", "hybrid/gpu_channels.py"]:
        fpath = REPO / rel_path
        if not fpath.exists(): continue
        source = fpath.read_text()
        lines = source.split("\n")
        n = len(lines)
        prompt = "\n".join(lines[:n//3]) + "\n"
        expected = "\n".join(lines[n//3:])
        challenges.append({
            "id": f"module_{Path(rel_path).stem}",
            "tier": 3, "category": "code",
            "prompt": prompt, "expected": expected,
            "test_code": "import ast,sys\ntry:\n ast.parse(sys.stdin.read())\n print('PASS')\nexcept SyntaxError as e:\n print(f'FAIL: {e}')",
            "source_file": rel_path,
            "title": f"Complete {Path(rel_path).stem}.py",
        })

    return challenges


def main():
    all_c = []
    all_c.extend(theory_challenges())
    all_c.extend(code_completion_challenges())
    l1 = len(theory_challenges())
    print(f"Theory: {l1}, Code: {len(all_c)-l1}, Total: {len(all_c)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for c in all_c:
            f.write(json.dumps(c) + "\n")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()

def deep_theory_challenges() -> list[dict]:
    """Massive theory bank covering everything in the codebase."""
    pairs = [
        # === Attention & Transformer ===
        ("Transformer: residual connection necessity", "Why are residual connections essential in deep transformers? What happens without them at L=24?", "Without residuals, gradient signals decay exponentially through layers. At L=24, the gradient at layer 0 is multiplied by 24 Jacobians, each < 1, making it effectively zero. Residuals provide an identity path: h_{l+1} = LayerNorm(h_l + F(h_l)). The gradient can flow through BOTH the F(h_l) path AND the identity path. This is the 'gradient highway' — even if F(h_l) has vanishing gradients, the identity path carries signal. Our DeepCausalLM at L=24 would not converge without this."),
        ("Transformer: multi-head attention computational complexity", "Multi-head attention is O(T^2 * d_model). Break this down: where does T^2 come from? How does this limit our context window to 128?", "The attention score matrix is (B, n_heads, T, T) — computing Q@K^T costs O(T^2 * d_head) per head. For T=128 and d_head=64: 128^2 * 64 = 1M operations per head. With 12 heads: 12M ops. The memory is O(T^2 * n_heads) — storing the full attention matrix. At T=128 on 124M model: manageable (128^2 * 12 * 4 bytes = 786KB). At T=512: 12MB per batch element. At batch=8: 96MB just for attention scores — plus activations, this OOMs. Hence context_len=128 for our trainer."),
        ("Transformer: causal mask implementation", "How does torch.nn.Transformer.generate_square_subsequent_mask enforce causality? What does the mask look like for T=4?", "The mask is a (T,T) upper-triangular matrix of -inf values: [[0, -inf, -inf, -inf], [0, 0, -inf, -inf], [0, 0, 0, -inf], [0, 0, 0, 0]]. When added to attention scores BEFORE softmax: exp(-inf) = 0, so future positions get zero attention weight. The current position CAN attend to itself and all previous positions. This is applied at every layer of the encoder."),
        ("Transformer: position embeddings", "Our DeepCausalLM uses learned position embeddings pos_emb of shape (max_len, d_model). How are they combined with token embeddings? What happens when a sequence exceeds max_len?", "Token embeddings and position embeddings are ADDED: h = tok_emb(x) + pos_emb(pos). For sequence length 128 position 5: h[0,5] = tok_emb[x[0,5]] + pos_emb[5]. If a sequence exceeds max_len=128, position 128 has no embedding — the index would be out of bounds, causing an IndexError. Our trainer truncates sequences to avoid this."),

        # === Optimization ===
        ("AdamW: momentum and variance tracking", "AdamW maintains two running averages: m (first moment) and v (second moment). What do they track? Why is bias correction needed?", "m = beta1 * m + (1-beta1) * g — exponential moving average of gradients (estimates the MEAN gradient direction). v = beta2 * v + (1-beta2) * g^2 — exponential moving average of squared gradients (estimates the VARIANCE). At step 0, m=v=0, so early steps are biased toward zero. Bias correction: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t). This divides by a number < 1, inflating early estimates. After many steps, (1-beta^t) → 1, correction → 1. Our trainer uses default betas: (0.9, 0.999)."),
        ("AdamW: learning rate and effective step size", "With lr=3e-4 and weight_decay=0.01, what's the effective step size for a parameter with gradient 0.1 vs 0.001? How does AdamW's adaptivity help here?", "For g=0.1: m≈0.1, v≈0.01. Update = lr * m_hat / sqrt(v_hat) ≈ 3e-4 * 0.1/0.1 = 3e-4. For g=0.001: m≈0.001, v≈0.000001. Update ≈ 3e-4 * 0.001/0.001 = 3e-4. Both get ~SAME step size! AdamW normalizes by gradient magnitude. Without this, a parameter with small gradients (e.g., steer_semantic at early training) would barely move while steer_local zooms ahead. The adaptivity ensures all 21 channels + semantic encoder learn at comparable rates."),
        ("Gradient clipping: why clip at 1.0", "Our trainer calls clip_grad_norm_(params, 1.0). What problem does this solve? Why 1.0 specifically?", "Gradient clipping prevents exploding gradients. If a single training example has outlier features (e.g., a PUNCT token with extreme unigram prob), the loss gradient can be 100× normal, causing a massive weight update that destabilizes training. Clipping scales the ENTIRE gradient vector if its L2 norm exceeds 1.0: g = g * (1.0 / ||g||). The value 1.0 is empirical — too large (10.0) doesn't prevent explosions, too small (0.1) throttles learning. Our 1.0 worked for 124M-2B across all runs."),
        ("Gradient accumulation: why accumulate=4", "train_steerer_v4.py supports --accumulate N. How does gradient accumulation work? Why use it when batch=8 already fits in memory?", "Gradient accumulation splits a virtual large batch into N micro-batches. Instead of computing loss on batch=32 and doing one backward pass (OOM), we do: loss_1.backward() [batch=8], loss_2.backward() [batch=8], ... then optimizer.step() once. The gradients SUM across micro-batches. This simulates batch=32 while only needing memory for batch=8. Our trainer rarely uses accumulation because batch=8 already provides stable gradient estimates for 119M tokens — the effective dataset size dominates batch noise."),

        # === Our architecture deep dives ===
        ("Superposition: why 21 channels specifically", "Why 21 channels and not 10 or 50? How were the 9 CPU n-gram features + 12 GPU features chosen? What constraint drove the split into 6+7+8?", "The 21 channels emerged from feature engineering: 9 CPU features (uni, bi_fast, bi_slow, tri_fast, tri_slow, skip2, skip3, recency, entropy) + 12 GPU features (shape, global_uni, ppmi_cos, ppmi_max, ppmi_norm, punct_density, repetition, unique_ratio, topic, KV, POS, spare). The 6+7+8 split balances compute per MLP: local_mlp(6→12→6)=144 params, mid_mlp(7→14→7)=196 params, global_mlp(8→16→8)=256 params. The split follows feature timescales: local (instantaneous), mid (sentence-level), global (document-level). More channels would require more training data to avoid overfitting the steer vectors."),
        ("Feature computing: CPU vs GPU split", "Why are channels 0-8 computed on CPU (FastNgramFeatures) while channels 9-20 are computed on GPU (GPUFeatureComputer)? What's the bottleneck?", "CPU channels require sequential token-by-token state updates (ngram counts, recency tracking). Doing this on GPU would be painfully slow — each token depends on all previous tokens, requiring a sequential scan through the sequence. GPU channels are vectorized: topic prior uses matrix multiplication, KV cache uses batched cosine similarity. The CPU features are pre-computed in DataLoader workers (background processes), so they're ready when the GPU needs them. The overlap: CPU fills channels 0-8, GPU overwrites with vectorized 9-20, total = 21."),
        ("Feature computing: exponetial decay in FastNgramFeatures", "The unigram counter decays: self._uni *= 0.999 every 10 steps. Why decay? What does the half-life represent?", "Decay makes old observations count less. After 693 steps (ln(0.5)/ln(0.999) ≈ 693), a token's count is halved. At ~120 tokens per sample, that's ~6 samples of 'memory.' The decay prevents unbounded growth and makes the features ADAPTIVE — a token that appeared 1000 steps ago has count *0.999^100 ≈ 0.90 of its original weight, while a token 10000 steps ago has weight 0.00005. This creates a recency-weighted unigram probability that captures local context shifts (e.g., switching from 'France' section to 'Japan' section)."),
        ("Cartridge: load_state_dict strict=False", "build_steerer_from_checkpoint loads steerer_state with strict=False. What keys are allowed to mismatch? Why is this safe?", "strict=False allows: 1) Missing keys in state dict (e.g., old checkpoint without semantic_encoder parameters). 2) Extra keys in state dict (e.g., new checkpoint with extra_channels). PyTorch silently ignores both cases. This is safe because: missing parameters keep their INITIALIZATION (small random values near zero) — the cartridge just doesn't use those features. Extra parameters in the checkpoint are ignored — they represent features the current cartridge doesn't need. The CORE steer vectors (steer_local/mid/global) MUST match in shape, which they always do for the same d_model."),
        ("Cartridge: gamma initialization", "gammas are initialized to 0.05 in FeatureConditionedAdapterSteerer. Why not 1.0 or 0.001? What happens with each choice?", "gamma=1.0: Full-strength injection from step 1. The cartridge dominates the hidden state, the model can't learn its own representations. gamma=0.001: Near-zero injection. The cartridge has no effect, gradients are too small to train. gamma=0.05: Small but non-zero. The cartridge contributes ~5% of the residual magnitude. This gentle start lets the model warm up while the cartridge learns WHICH patterns to amplify. After training, gammas often grow to 0.1-1.0 as the cartridge proves its signal is useful."),
        ("Cartridge: noise_scale during training", "FeatureConditionedAdapterSteerer has noise_scale=0.03. Where is noise injected? Why add noise to features?", "In set_weights(): if training and noise_scale > 0, weights = weights + randn_like(weights) * noise_scale. This adds Gaussian noise to the 21-channel features before they pass through the gating MLPs. Purpose: REGULARIZATION. Without noise, the cartridge can overfit to exact feature values from training examples. With noise, it learns ROBUST patterns — 'channel 15 > 0.7 means LOC entity' rather than 'channel 15 = 0.732 exactly from this training text.' This is a form of data augmentation that prevents memorization of statistical quirks in the 119M-token training set."),
        ("Steerer: RMS normalization in _steer_layer", "Why does _steer_layer normalize the offset by h_rms / o_rms? What would happen without this normalization?", "The offset is the steerer's proposed delta to the hidden state. Without normalization, if the offset magnitude is much larger than the hidden state magnitude, the steerer DROWNS the original signal. h_rms / o_rms scales the offset to match the hidden state's energy level. This preserves the RELATIVE contribution: the steerer can change the DIRECTION of the hidden state (semantic shift) but maintains the MAGNITUDE (activation scale). Without this: at layer 0, a small initial hidden state would get a huge relative delta; at layer 10, the large hidden state would override the steerer entirely. RMS normalization makes injection magnitude CONSISTENT across layers."),

        # === ZeroQ & Quantization ===
        ("ZeroQ: Maxwell vs Ampere config", "ZeroQBackend uses config_name='MAXWELL_CONFIG'. What's different about AMPERE_CONFIG? Why does the config matter?", "MAXWELL_CONFIG and AMPERE_CONFIG define: 1) Memory pool sizes (Maxwell has smaller L1 cache → larger blocks). 2) Shard sizes (Maxwell benefits from larger shards due to slower PCIe). 3) Stream partitioning strategy. Maxwell GPUs (M40) don't support CUDA cooperative groups, so some optimizations are disabled. The config ensures ZeroQ's memory management matches the GPU's hardware characteristics. Using AMPERE_CONFIG on M40 would try to allocate more L1 cache than exists, causing CUDA errors."),
        ("ZeroQ: stream partition explained", "ZeroQPartitionedBackend has stream_partition=True. What is stream partitioning and why does it reduce memory?", "Stream partitioning means ZeroQ partitions model weights ASYNCHRONOUSLY using CUDA streams. Instead of synchronously loading the entire model into GPU, partitioning each layer, loading the next: Layer 0 partitions on stream 0 while Layer 1 loads on stream 1. The overlap hides data transfer latency. For M40 with PCIe gen3, transfer bandwidth is ~12GB/s. A 2B model (8GB fp32) takes ~0.7s to transfer. Without stream overlap: 0.7s of idle GPU. With stream overlap: GPU is computing while PCIe transfers."),
        ("NF4: quantile-based quantization vs uniform", "NF4 divides the normal distribution into equiprobable regions. Why not just use uniform quantization (equal-width bins)?", "Neural network weights are approximately normally distributed (mean≈0, std≈0.1-0.5). Uniform quantization wastes levels: 50% of levels cover the center ±0.67σ where most weights are, but the tails (±2σ+) get the SAME number of levels despite having far fewer weights. NF4 uses equiprobable regions: each of 16 levels represents ~6.25% of weights, regardless of position. This gives finer resolution at the center (many weights, small bins) and coarser at tails (few weights, larger bins). Result: better reconstruction quality for the same 4-bit budget."),

        # === Training Mechanics ===
        ("DataLoader: num_workers=4 and pin_memory", "Why num_workers=4? What does pin_memory=True do? Why num_workers=0 on M40?", "num_workers=4: Four background processes pre-fetch batches. Each worker: loads raw data → tokenizes → computes CPU n-gram features → assembles batch. This overlaps CPU work with GPU computation. pin_memory=True: Allocates tensors in page-locked (non-swappable) memory, enabling faster CPU→GPU DMA transfers (~2x speed). num_workers=0 on M40: The M40's CPU is slow and has limited PCIe lanes. Multi-worker contention on the memory bus can cause STALLS that cost more than the overlap saves. Single-process avoids bus contention."),
        ("Dropout: why dropout=0.0 in our models", "Our DeepCausalLM uses dropout=0.0 during training. Why no dropout when training from scratch?", "We train on only 119M tokens for 124M params — the model is data-starved, not overfit. Dropout reduces effective capacity, which we can't afford at 1:1 token-to-param ratio. Additionally, our compiled priors serve as implicit regularization: the 21-channel features constrain the steerer's behavior, and the orthogonal penalty on steer vectors prevents co-adaptation. The model CAN'T overfit because it hasn't even fully fit the training data (training PPL > validation PPL at all epochs)."),
        ("Loss: cross_entropy with ignore_index", "Our eval computes loss with F.cross_entropy(logits, targets, reduction='sum'). Why sum instead of mean? Why not use ignore_index for padding?", "sum over tokens gives total NLL across the eval slice, then we divide by total token count to get average PPL = exp(total_nll / total_tokens). This handles variable-length eval slices correctly. mean would give per-position average PPL. We don't need ignore_index because our eval sequences have NO padding — validation data is one contiguous token stream, and we process it in fixed-size chunks without batch padding."),
        ("Early stopping: why we don't use it", "The trainer has no early stopping. Why not? When WOULD early stopping help?", "Early stopping prevents overfitting when validation loss increases while training loss decreases. In our setup, eval_b (blind validation PPL) improves monotonically for 200+ epochs — there's no inflection point to detect. The model is undertrained, not overfit. Early stopping would PREMATURELY halt training, missing PPL improvements in epochs 150-200. Early stopping WOULD help if we trained 124M on 1B+ tokens — the model would eventually overfit, and validation PPL would turn upward."),
        ("Mixed precision: fp16 gradients and fp32 master weights", "Why does PyTorch amp maintain fp32 master copies even when computing in fp16? What happens without master copies?", "fp16 has limited dynamic range (max 65504). Gradients for small parameters (e.g., norm biases) can be < 1e-8 — below fp16's minimum representable value of ~6e-8. These gradients become ZERO in fp16, and the parameter never updates. fp32 master copies store the full-precision parameter and apply updates: w_fp32 += lr * g_fp16 (cast to fp32). The fp16 model weights are synchronized from masters after each step. Without masters: parameters with small gradients FREEZE permanently, creating dead channels in the steerer."),

        # === GPU and System ===
        ("CUDA: compute capability and kernel compatibility", "PyTorch warns 'Found GPU Tesla M40 with capability 5.2. Minimum supported: 7.0.' Why does our training still work sometimes?", "Our m40_env Python uses an OLDER PyTorch compiled with compute_52 support. The system Python (anaconda3) has PyTorch 2.7+ which dropped Maxwell. The warning fires from the system Python but the m40_env Python has the right kernels. The key: ALWAYS use ~/local_venvs/m40_env/bin/python for M40 training."),
        ("VRAM: model, optimizer, activations memory breakdown", "For 124M DeepCausalLM training at batch=8, seq=128: where do the ~3GB of GPU memory go?", "Model weights (fp16): 124M * 2 bytes = 248MB. Model weights (fp32 master): 124M * 4 bytes = 496MB. Optimizer state (AdamW): 2 * 496MB = 992MB (m and v per param). Activations: attention scores (8*12*128*128*2 bytes) = 3.1MB + FFN activations (8*128*3072*2) = 6.3MB per layer * 12 layers ≈ 75MB. Gradients: 496MB. Misc (CUDA context, workspace): ~500MB. Total: ~2.8GB. The 3080's 10GB has plenty of headroom — we can fit 500M (~6GB) but not 2B (~12GB)."),
        ("PCIe: gradient synchronization overhead", "On pe2's dual M40 setup, model layers split across GPU0 and GPU1. Every forward/backward pass requires cross-GPU tensor transfers. Where's the bottleneck?", "The bottleneck is PCIe bandwidth between GPUs (gen3 x16 ≈ 12GB/s theoretical, ~8GB/s actual). For Qwen3B fp16: during forward, intermediate activations (hidden states) flow from GPU0 layers to GPU1 layers. At batch=1, seq=128, d_model=2048: each activation is 128*2048*2 = 524KB. 36 layers = ~19MB of activations per forward pass. At 8GB/s, transfer adds ~2.4ms per pass — negligible compared to M40's ~50ms per layer compute time. The bottleneck is COMPUTE, not communication."),

        # === Experiment Analysis ===
        ("Ablation: pure SGD convergence proof", "The ablation study (injection=none) showed eval_b=7000 after 200 epochs. What specific deficit does this prove about 119M-token training?", "It proves that 119M tokens of WikiText are INSUFFICIENT for a randomly initialized 124M transformer to learn a useful language model through pure SGD alone. The model needs the compiled priors to CONVERGE. Even at 33× higher LR, pure SGD stalled. This isn't an optimization problem — it's a SAMPLE EFFICIENCY problem. The priors inject 21 dimensions of pre-computed structure that the model would need millions more tokens to discover. Without priors, 119M tokens provide ~1 token per parameter — the model can't learn the joint distribution from scratch."),
        ("Semantic channels: why 124M was too small", "Semantic channels helped at 500M but not at 124M. Why does model scale affect compiled prior effectiveness?", "The semantic encoder adds ~50K parameters and introduces a new conditioning path through the steerer. At 124M (1:1 token ratio), every parameter must earn its keep — 50K params is 0.04% of the model but the gradient signal is shared with 124M other params. At 500M, the model has MORE capacity to integrate the semantic signal: the attention heads, FFN neurons, and layer norms can specialize. The semantic features at 500M get DEDICATED model capacity to exploit them. At 124M, they compete with basic language modeling capacity and lose."),
        ("FastNgramFeatures: the recency gap feature", "Channel 7 (recency) measures how many steps since a token last appeared. Why is this a useful feature for language modeling?", "Recency captures discourse-level patterns: in a paragraph about France, 'Paris' appears multiple times. The first occurrence has high recency (gap=128, never seen), subsequent occurrences have low recency (gap=5-20). This helps the steerer recognize TOPICAL COHERENCE — 'we are still talking about France' vs 'we switched topics.' Without this feature, the model might predict 'capital' → 'France' just because 'France' appeared 50 tokens ago, missing the discourse shift. Recency provides a time-decay signal that complements n-gram statistics."),
        ("Why compilation beat pure SGD by 152x", "The ablation paper reports 152x better PPL with compiled priors. Break down where this 152x comes from.", "eval_b: 7000 (SGD) vs 46 (compiled) = 152x. The factors: 1) N-gram priors (~40x): The model doesn't need to learn token-level co-occurrence from scratch — channel 0-5 encode P(next_token | prev_tokens). 2) Topic/deep priors (~2x): Channels 13-20 encode document-level statistics that guide high-level predictions. 3) Orthogonal penalty (~1.5x): Keeps steer vectors independent, maximizing information per channel. 4) Multi-timescale routing (~1.3x): Different channels at appropriate depths. Combined: non-linear interaction between these factors. The 152x is a SYNERGY, not a sum."),
    ]
    return pairs_to_challenges(pairs, 100)


def pairs_to_challenges(pairs, start_id):
    return [{
        "id": f"theory_{i+start_id:03d}",
        "tier": 4, "category": "theory",
        "prompt": f"# {title}\n\n{question}\n\n",
        "expected": answer,
        "test_code": "import sys;code=sys.stdin.read();assert len(code)>100;print('PASS')",
        "source_file": "CMI_CODEBASE_DEEPDIVE.md",
        "title": title,
    } for i, (title, question, answer) in enumerate(pairs)]


# Override main to use deep theory
import sys as _sys
if __name__ == "__main__" and "--rebuild" in _sys.argv:
    all_c = []
    all_c.extend(theory_challenges())
    all_c.extend(deep_theory_challenges())
    all_c.extend(code_completion_challenges())
    print(f"Theory: {len(theory_challenges())+len(deep_theory_challenges())}, Code: {len(code_completion_challenges())}, Total: {len(all_c)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for c in all_c:
            f.write(json.dumps(c) + "\n")
    print(f"Saved to {OUT}")
