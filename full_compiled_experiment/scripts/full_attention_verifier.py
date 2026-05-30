"""
Phase 3a: Full attention fidelity verifier.

Extracts ALL Q, K, V, O weights from Qwen2.5-1.5B layer 0,
recomputes the full multi-head attention manually, and compares
against the actual model's attention output.

Expected: >0.99 cosine similarity (numerical precision limits only).
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ucn.stdlib.loader import save_stdlib_json, save_weight_tensor
from ucn.stdlib.schema import BehaviorMeta, MathDef, PrimitiveEntry


def get_attention_weights(model):
    l0_attn = model.model.layers[0].self_attn
    return {
        "W_q": l0_attn.q_proj.weight.data.clone().float(),
        "b_q": l0_attn.q_proj.bias.data.clone().float() if l0_attn.q_proj.bias is not None else None,
        "W_k": l0_attn.k_proj.weight.data.clone().float(),
        "b_k": l0_attn.k_proj.bias.data.clone().float() if l0_attn.k_proj.bias is not None else None,
        "W_v": l0_attn.v_proj.weight.data.clone().float(),
        "b_v": l0_attn.v_proj.bias.data.clone().float() if l0_attn.v_proj.bias is not None else None,
        "W_o": l0_attn.o_proj.weight.data.clone().float(),
        "b_o": l0_attn.o_proj.bias.data.clone().float() if l0_attn.o_proj.bias is not None else None,
    }


def manual_attention(hidden_states, weights, position_ids, rotary_emb, device):
    """
    Reproduce Qwen2.5-1.5B layer 0 attention exactly.
    """
    W_q = weights["W_q"].to(device)
    W_k = weights["W_k"].to(device)
    W_v = weights["W_v"].to(device)
    W_o = weights["W_o"].to(device)

    B, T, D = hidden_states.shape
    n_heads = 12
    n_kv_heads = 2
    head_dim = 128
    n_groups = n_heads // n_kv_heads

    hs = hidden_states.float().to(device)

    q = hs @ W_q.T
    if weights.get("b_q") is not None:
        q = q + weights["b_q"].to(device)
    k = hs @ W_k.T
    if weights.get("b_k") is not None:
        k = k + weights["b_k"].to(device)
    v = hs @ W_v.T
    if weights.get("b_v") is not None:
        v = v + weights["b_v"].to(device)

    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_kv_heads, head_dim).transpose(1, 2)

    cos, sin = rotary_emb(
        torch.zeros(1, T, head_dim, device=device),
        position_ids.to(device),
    )

    cos = cos.to(dtype=torch.float32)
    sin = sin.to(dtype=torch.float32)
    q = q.float()
    k = k.float()

    q_rot = torch.zeros_like(q)
    k_rot = torch.zeros_like(k)

    q_rot[..., :head_dim // 2] = (
        q[..., :head_dim // 2] * cos[..., :head_dim // 2]
        - q[..., head_dim // 2:] * sin[..., :head_dim // 2]
    )
    q_rot[..., head_dim // 2:] = (
        q[..., :head_dim // 2] * sin[..., :head_dim // 2]
        + q[..., head_dim // 2:] * cos[..., :head_dim // 2]
    )

    k_rot[..., :head_dim // 2] = (
        k[..., :head_dim // 2] * cos[..., :head_dim // 2]
        - k[..., head_dim // 2:] * sin[..., :head_dim // 2]
    )
    k_rot[..., head_dim // 2:] = (
        k[..., :head_dim // 2] * sin[..., :head_dim // 2]
        + k[..., head_dim // 2:] * cos[..., :head_dim // 2]
    )

    k_rep = k_rot.unsqueeze(2).expand(B, n_kv_heads, n_groups, T, head_dim)
    k_rep = k_rep.reshape(B, n_heads, T, head_dim)
    v_rep = v.unsqueeze(2).expand(B, n_kv_heads, n_groups, T, head_dim)
    v_rep = v_rep.reshape(B, n_heads, T, head_dim)

    scaling = head_dim ** -0.5
    attn_logits = (q_rot @ k_rep.transpose(-2, -1)) * scaling

    causal_mask = torch.triu(
        torch.full((T, T), float("-inf"), device=device),
        diagonal=1,
    )
    attn_logits = attn_logits + causal_mask

    attn_weights = F.softmax(attn_logits.float(), dim=-1)

    attn_output = attn_weights @ v_rep

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, T, D)
    attn_output = attn_output @ W_o.T
    if weights.get("b_o") is not None:
        attn_output = attn_output + weights["b_o"].to(device)

    return attn_output.to(dtype=hidden_states.dtype)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen2.5-1.5B (float32, eager attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

    rotary_emb = model.model.rotary_emb
    l0_attn = model.model.layers[0].self_attn

    print("Extracting Q/K/V/O weights from layer 0...")
    weights = get_attention_weights(model)
    for k, v in weights.items():
        if v is not None:
            print(f"  {k}: {list(v.shape)}")
        else:
            print(f"  {k}: None")

    test_prompts = [
        "The cat sat on the mat and looked around.",
        "Machine learning enables computers to learn from data.",
        "The capital of France is Paris, a city known for its art.",
        "Python is a high-level programming language for data science.",
        "The quick brown fox jumps over the lazy dog near the river.",
        "Deep learning models require large amounts of data.",
        "The Earth orbits the Sun at about 93 million miles.",
        "Neural networks consist of interconnected layers of nodes.",
        "Water boils at 100 degrees Celsius at standard pressure.",
        "Shakespeare wrote Hamlet and Romeo and Juliet.",
    ]

    attn_inputs = []
    attn_outputs_real = []

    def pre_hook(module, args, kwargs):
        if "hidden_states" in kwargs:
            attn_inputs.append(kwargs["hidden_states"].detach().clone())

    def post_hook(module, args, output):
        if isinstance(output, tuple):
            attn_outputs_real.append(output[0].detach().clone())
        else:
            attn_outputs_real.append(output.detach().clone())

    handle_pre = l0_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    handle_post = l0_attn.register_forward_hook(post_hook)

    print(f"\nRunning {len(test_prompts)} test prompts...")
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    handle_pre.remove()
    handle_post.remove()

    print(f"\nComparing manual attention vs real attention...")
    print(f"{'='*70}")

    all_cosines = []
    all_mses = []

    for i, (hs, real_out) in enumerate(zip(attn_inputs, attn_outputs_real)):
        T = hs.shape[1]
        position_ids = torch.arange(T).unsqueeze(0)

        with torch.no_grad():
            manual_out = manual_attention(hs, weights, position_ids, rotary_emb, device)

            real_out_cpu = real_out.float().cpu()
            manual_out_cpu = manual_out.float().cpu()

            real_out_flat = real_out_cpu.reshape(-1)
            manual_out_flat = manual_out_cpu.reshape(-1)

            cos_sim = F.cosine_similarity(
                real_out_flat.unsqueeze(0),
                manual_out_flat.unsqueeze(0),
                dim=-1,
            ).item()

            mse = F.mse_loss(manual_out_cpu, real_out_cpu).item()

            all_cosines.append(cos_sim)
            all_mses.append(mse)

            # Per-token fidelity
            per_token_cos = []
            for t in range(T):
                c = F.cosine_similarity(
                    manual_out_cpu[0, t].unsqueeze(0),
                    real_out_cpu[0, t].unsqueeze(0),
                    dim=-1,
                ).item()
                per_token_cos.append(c)

            prompt_preview = test_prompts[i][:50]
            print(f"  Prompt {i+1}: '{prompt_preview}...'")
            print(f"    Overall cosine: {cos_sim:.8f}")
            print(f"    Overall MSE:    {mse:.8f}")
            print(f"    Per-token:      {[f'{c:.6f}' for c in per_token_cos]}")

    avg_cos = sum(all_cosines) / len(all_cosines)
    avg_mse = sum(all_mses) / len(all_mses)
    min_cos = min(all_cosines)
    max_cos = max(all_cosines)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Mean cosine similarity: {avg_cos:.8f}")
    print(f"  Min  cosine similarity: {min_cos:.8f}")
    print(f"  Max  cosine similarity: {max_cos:.8f}")
    print(f"  Mean MSE:              {avg_mse:.8f}")

    if avg_cos > 0.999:
        print(f"\n  VERDICT: PERFECT FIDELITY (>0.999 cosine)")
        print(f"  The full attention computation reproduces the model exactly.")
    elif avg_cos > 0.99:
        print(f"\n  VERDICT: NEAR-PERFECT FIDELITY (>0.99 cosine)")
    elif avg_cos > 0.90:
        print(f"\n  VERDICT: HIGH FIDELITY (>0.90 cosine)")
    else:
        print(f"\n  VERDICT: Fidelity needs investigation")

    print(f"\n  Comparison with V*O-only approach:")
    print(f"    V*O-only (single head):  0.18 cosine")
    print(f"    V*O-only (all 12 heads): 0.25 cosine")
    print(f"    Full attention:          {avg_cos:.4f} cosine")
    print(f"    Improvement:             {avg_cos - 0.25:.4f} absolute gain")

    # Save weights for later use in UCN
    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "full_attention_verifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, v in weights.items():
        if v is not None:
            save_weight_tensor(v, out_dir / f"l0_{k}.pt")

    report = {
        "method": "full_attention",
        "model": "Qwen2.5-1.5B",
        "layer": 0,
        "n_heads": 12,
        "n_kv_heads": 2,
        "head_dim": 128,
        "avg_cosine": avg_cos,
        "min_cosine": min_cos,
        "max_cosine": max_cos,
        "avg_mse": avg_mse,
        "n_test_prompts": len(test_prompts),
    }

    with open(out_dir / "full_attention_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print(f"\nWeights saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
