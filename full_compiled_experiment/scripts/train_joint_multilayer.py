"""
Path A: Joint end-to-end MetaCompiler training for multi-layer fidelity.
Runs on pe3 (2× M40 12GB) with per-token cosine loss, WikiText-103 data,
cosine LR schedule, gradient accumulation, and checkpointing.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python3 scripts/train_joint_multilayer.py \
    --steps 2000 --lr 1e-4 --n-calib 340 --calib-max-tokens 96
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from ucn.dsl.ast import MatrixRef, Program, Transform
from ucn.frontend.meta_compiler import MetaCompiler
from ucn.runtime.executor import UCNExecutor, rms_norm
from ucn.backend.codegen.reference import ReferenceBackend


def load_wikitext_prompts(n_prompts=340, max_tokens=96, seed=42):
    """Load WikiText-103 passages as training prompts."""
    data_path = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "wikitext_gpt2" / "train_ids.pt"
    if not data_path.exists():
        data_path = Path.home() / "deepseek_experiments" / "artifacts" / "wikitext_gpt2" / "train_ids.pt"
    if not data_path.exists():
        print(f"  WikiText not found at {data_path}, using fallback prompts")
        return _fallback_prompts(n_prompts)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ids = torch.load(data_path)

    prompts = []
    torch.manual_seed(seed)
    start = 0
    while len(prompts) < n_prompts and start < len(ids) - max_tokens:
        end = min(start + max_tokens, len(ids))
        chunk = ids[start:end]
        decoded = tokenizer.decode(chunk.tolist(), skip_special_tokens=True)
        decoded = decoded.strip()
        if len(decoded.split()) >= 10:
            prompts.append(decoded)
        start += torch.randint(50, 200, (1,)).item()
        if start >= len(ids) // 2:
            start = 0
            torch.manual_seed(seed + len(prompts))

    print(f"  Loaded {len(prompts)} WikiText prompts")
    return prompts[:n_prompts]


def _fallback_prompts(n):
    return [
        "The sun rises in the east every morning without fail.",
        "Computers process information using binary code and logic.",
        "The human brain is one of the most complex organs known.",
        "Water is essential for all known forms of life on Earth.",
        "Music is a universal language that transcends boundaries.",
        "Mathematics is the foundation of scientific disciplines.",
        "The industrial revolution transformed manufacturing globally.",
        "Climate patterns have shifted significantly in recent decades.",
        "Artificial intelligence research has accelerated rapidly since 2012.",
        "The periodic table organizes elements by atomic structure.",
        "Shakespeare wrote tragedies that explore human nature deeply.",
        "The speed of light is constant in all reference frames.",
        "Quantum mechanics describes behavior at subatomic scales.",
        "The Roman Empire influenced law and governance for centuries.",
        "Photosynthesis converts sunlight into chemical energy for plants.",
        "Neural networks learn patterns from large datasets iteratively.",
        "The Amazon rainforest produces significant oxygen for the planet.",
        "DNA replication ensures genetic information passes to offspring.",
        "The printing press revolutionized information distribution in Europe.",
        "Fossil fuel consumption has driven economic growth since 1800.",
    ][:n]


def build_stdlib_and_programs(model, target_layers):
    from ucn.decompile.mlp_decomposer import extract_full_mlp_weights_lr

    cos, sin = model.model.rotary_emb(
        torch.zeros(1, 256, 128, device=model.device),
        torch.arange(256, device=model.device).unsqueeze(0)
    )

    stdlib = {}
    for l in target_layers:
        a = model.model.layers[l].self_attn
        stdlib[f"a_L{l}"] = {
            "operator_type": "multihead_attention",
            "W_q": a.q_proj.weight.float().cpu(),
            "W_k": a.k_proj.weight.float().cpu(),
            "W_v": a.v_proj.weight.float().cpu(),
            "W_o": a.o_proj.weight.float().cpu(),
            "b_q": a.q_proj.bias.float().cpu() if a.q_proj.bias is not None else None,
            "b_k": a.k_proj.bias.float().cpu() if a.k_proj.bias is not None else None,
            "b_v": a.v_proj.bias.float().cpu() if a.v_proj.bias is not None else None,
            "cos": cos.cpu(), "sin": sin.cpu(),
            "n_heads": 12, "n_kv_heads": 2, "head_dim": 128,
        }
        mw = extract_full_mlp_weights_lr(model, l, rank=128)
        stdlib[f"m_L{l}"] = {
            "operator_type": "sparse_down_projection_lr",
            "gate_weight": mw["gate_weight"], "gate_bias": mw["gate_bias"],
            "up_weight": mw["up_weight"], "down_weight": mw["down_weight"],
            "U_r": mw["U_r"], "V_r": mw["V_r"], "top_k": 1024,
        }
    return stdlib


def run_ucn_multilayer(emb, executor, backend, meta_compilers, target_layers, norm_weights, collect_layer_outputs=False):
    """Forward pass through all UCN layers. Returns (final_h, [per_layer_h] if collect)."""
    h = emb.clone()
    per_layer = [] if collect_layer_outputs else None

    for idx, l in enumerate(target_layers):
        ln_attn = norm_weights[l]["pre_attn"]
        ln_mlp = norm_weights[l]["pre_mlp"]

        hn = rms_norm(h, ln_attn)
        pa = Program()
        pa.add_stmt("y", Transform("x", MatrixRef("stdlib", f"a_L{l}")))
        h = h + executor.execute_raw(pa, {"x": hn}).get("y", torch.zeros_like(h))

        hn = rms_norm(h, ln_mlp)
        stdlib_names = [f"m_L{tl}" for tl in target_layers]
        mc_emb = hn.clone()

        mlp_out = meta_compilers[idx].synthesize_soft_forward(
            mc_emb, stdlib_names=stdlib_names, executor=backend, temperature=1.0
        )
        h = h + mlp_out

        if collect_layer_outputs:
            per_layer.append(h.clone())

    if collect_layer_outputs:
        return h, per_layer
    return h


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-calib", type=int, default=340)
    parser.add_argument("--calib-max-tokens", type=int, default=96)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    steps = args.steps
    lr = args.lr
    print(f"Config: device={device}, steps={steps}, lr={lr}, accum={args.accum}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen2.5-1.5B (fp16, sdpa for teacher data)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B", trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

    print("Loading Qwen2.5-1.5B (fp32, eager, CPU) for stdlib weights...")
    model_eager = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B", trust_remote_code=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to("cpu")
    model_eager.eval()

    target_layers = [0, 4, 8]

    print("Building stdlib...")
    stdlib = build_stdlib_and_programs(model_eager, target_layers)
    executor = UCNExecutor(d_model=1536, stdlib_weights=stdlib, device=device, dtype=torch.float32)
    backend = ReferenceBackend(stdlib_weights=stdlib, device=device, dtype=torch.float32)

    norm_weights = {}
    for l in target_layers:
        norm_weights[l] = {
            "pre_attn": model_eager.model.layers[l].input_layernorm.weight.data.clone().to(device),
            "pre_mlp": model_eager.model.layers[l].post_attention_layernorm.weight.data.clone().to(device),
        }

    print(f"Loading {args.n_calib} WikiText prompts...")
    all_prompts = load_wikitext_prompts(n_prompts=args.n_calib, max_tokens=args.calib_max_tokens)
    n_val = min(60, len(all_prompts) // 5)
    n_train = min(len(all_prompts) - n_val, 340)
    train_prompts = all_prompts[:n_train]
    val_prompts = all_prompts[n_train:n_train + n_val]
    print(f"  Train: {len(train_prompts)}, Val: {len(val_prompts)}")

    def collect_teacher_data(prompts):
        teacher_targets = {l: [] for l in target_layers}
        train_embeddings = []
        hooks = []

        def make_hook(l):
            def hook(module, input, output):
                teacher_targets[l].append((output[0] if isinstance(output, tuple) else output).detach().cpu())
            return hook

        for l in target_layers:
            hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

        for prompt in prompts:
            inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.calib_max_tokens).to(device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
                train_embeddings.append(out.hidden_states[0].cpu())

        for h in hooks:
            h.remove()

        return train_embeddings, teacher_targets

    print("Collecting teacher data...")
    train_emb, train_teacher_list = collect_teacher_data(train_prompts)
    val_emb, val_teacher_list = collect_teacher_data(val_prompts)

    print(f"  Teacher data: {len(train_emb)} prompts")

    print(f"Instantiating {len(target_layers)} MetaCompilers (GPU)...")
    meta_compilers = [
        MetaCompiler(d_model=1536, n_templates=4, max_params=4, d_latent=64, n_layers=1, device=device)
        for _ in target_layers
    ]
    for i, mc in enumerate(meta_compilers):
        print(f"  MC[{target_layers[i]}]: {sum(p.numel() for p in mc.trainable_parameters())} params")

    all_params = []
    for mc in meta_compilers:
        all_params.extend(mc.trainable_parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    warmup_steps = min(50, steps // 4)

    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "joint_training"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "steps": steps, "lr": lr, "accum": args.accum,
        "n_train": n_train, "n_val": n_val,
        "max_tokens": args.calib_max_tokens,
        "target_layers": target_layers,
        "model": "Qwen2.5-1.5B",
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nJoint training ({steps} steps, lr={lr}, accum={args.accum})...")
    history = []
    best_val_score = -1.0
    best_state = None

    for step in range(steps):
        train_idx = step % len(train_emb)
        emb = train_emb[train_idx].clone()

        for mc in meta_compilers:
            mc.train()

        h = run_ucn_multilayer(
            emb.to(device), executor, backend, meta_compilers,
            target_layers, norm_weights, collect_layer_outputs=False
        )

        loss = 0.0
        for l_idx, l in enumerate(target_layers):
            teacher_prompt = train_teacher_list[l][train_idx].to(device)
            h_flat = h.reshape(-1, h.shape[-1])
            n_t = min(h_flat.shape[0], teacher_prompt.shape[0])
            loss = loss - F.cosine_similarity(h_flat[:n_t], teacher_prompt[:n_t], dim=-1).mean()

        loss = loss / len(target_layers)
        reg = 1e-4 * sum(p.abs().sum() for p in all_params)
        loss = loss + reg

        loss = loss / args.accum
        loss.backward()

        if (step + 1) % args.accum == 0:
            lr_scale = min(1.0, (step + 1) / max(warmup_steps, 1))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * lr_scale

            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            history.append({"step": step + 1, "loss": float(loss.item() * args.accum),
                            "lr": float(optimizer.param_groups[0]["lr"])})

        if step == 0 or (step + 1) % 100 == 0 or step == steps - 1:
            print(f"  Step {step+1:5d}/{steps}: loss={float(loss.item() * args.accum):.6f}  lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)

        if (step + 1) % 200 == 0 or step == steps - 1:
            for mc in meta_compilers:
                mc.eval()

            val_scores = []
            for l_idx, l in enumerate(target_layers):
                val_score_sum = 0.0
                n_samples = min(10, len(val_emb))
                for vi in range(n_samples):
                    with torch.no_grad():
                        h_val = run_ucn_multilayer(
                            val_emb[vi].to(device), executor, backend, meta_compilers,
                            target_layers, norm_weights, collect_layer_outputs=False
                        )
                        teacher_prompt = val_teacher_list[l][vi].to(device)
                        h_flat = h_val.reshape(-1, h_val.shape[-1])
                        n_t = min(h_flat.shape[0], teacher_prompt.shape[0])
                        val_score_sum += float(F.cosine_similarity(
                            h_flat[:n_t], teacher_prompt[:n_t], dim=-1
                        ).mean().item())
                avg_val = val_score_sum / max(n_samples, 1)
                val_scores.append(avg_val)

            avg_val_score = sum(val_scores) / len(val_scores)
            print(f"    Val cosine: avg={avg_val_score:.4f}  per_layer={[f'{s:.3f}' for s in val_scores]}", flush=True)

            if avg_val_score > best_val_score:
                best_val_score = avg_val_score
                best_state = [mc.state_dict() for mc in meta_compilers]
                for idx, mc in enumerate(meta_compilers):
                    torch.save(mc.state_dict(), out_dir / f"best_MC_L{target_layers[idx]}.pt")
                torch.save({"history": history, "step": step + 1, "val_score": avg_val_score, "val_per_layer": val_scores},
                           out_dir / "best.pt")

            for mc in meta_compilers:
                mc.train()

    print(f"\n{'='*50}")
    print(f"Training complete. Best val score: {best_val_score:.4f}")
    if best_state is not None:
        for idx, mc in enumerate(meta_compilers):
            mc.load_state_dict(best_state[idx])
            mc.eval()

    print("\nFinal evaluation:")
    for mc in meta_compilers:
        mc.eval()
    eval_prompt = val_prompts[0] if val_prompts else train_prompts[0]
    inp = tokenizer(eval_prompt, return_tensors="pt", truncation=True, max_length=args.calib_max_tokens).to(device)

    real_hs = {}
    def hk(l):
        def hook(module, input, output):
            real_hs[l] = (output[0] if isinstance(output, tuple) else output).detach().cpu()
        return hook
    eval_hooks = [model.model.layers[l].register_forward_hook(hk(l)) for l in target_layers]
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
        emb_eval = out.hidden_states[0].cpu()
    for h in eval_hooks:
        h.remove()

    h = emb_eval.clone()
    per_layer = {}
    for idx, l in enumerate(target_layers):
        ln_a = norm_weights[l]["pre_attn"]
        ln_m = norm_weights[l]["pre_mlp"]

        hn = rms_norm(h, ln_a)
        pa = Program()
        pa.add_stmt("y", Transform("x", MatrixRef("stdlib", f"a_L{l}")))
        h = h + executor.execute_raw(pa, {"x": hn}).get("y", torch.zeros_like(h))

        hn = rms_norm(h, ln_m)
        mc = meta_compilers[idx]
        mlp_out = mc.synthesize_soft_forward(
            hn.clone(), stdlib_names=[f"m_L{tl}" for tl in target_layers],
            executor=backend, temperature=0.1
        )
        h = h + mlp_out

        c = float(F.cosine_similarity(
            h.reshape(-1, h.shape[-1]), real_hs[l].reshape(-1, h.shape[-1]), dim=-1
        ).mean().item())
        per_layer[l] = c

    avg = sum(per_layer.values()) / len(per_layer)
    for l in target_layers:
        print(f"    Layer {l}: {per_layer[l]:.6f}")
    print(f"  Average: {avg:.6f}")
    print(f"  Baseline (no training): 0.340")
    print(f"  Improvement: {avg - 0.340:+.4f}")

    torch.save({"history": history, "per_layer": per_layer, "avg": avg,
                 "best_val_score": best_val_score},
               out_dir / "final.pt")
    for idx, mc in enumerate(meta_compilers):
        torch.save(mc.state_dict(), out_dir / f"final_MC_L{target_layers[idx]}.pt")

    print(f"\nSaved to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
