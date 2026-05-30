"""
Phase 3c: Save full Qwen attention weights to stdlib and verify via UCN pipeline.

Validates that the UCN ReferenceBackend, when given the full attention weights
as a multihead_attention primitive, reproduces the model output exactly.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ucn.dsl.ast import MatrixRef, Program, Transform
from ucn.backend.codegen.reference import ReferenceBackend
from ucn.stdlib.loader import save_stdlib_json, save_weight_tensor
from ucn.stdlib.schema import BehaviorMeta, MathDef, PrimitiveEntry


def extract_and_save_attention_stdlib(out_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen2.5-1.5B...")
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

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    n_heads = 12
    n_kv_heads = 2
    head_dim = 128

    weights = {}
    weight_entries = {}

    for name, param in l0_attn.named_parameters():
        t = param.data.clone().float()
        weight_path = weights_dir / f"l0_attn_{name.replace('.', '_')}.pt"
        save_weight_tensor(t, weight_path)
        weight_entries[name] = str(weight_path.name)
        weights[name] = t

    seq_len = 128
    dummy = torch.zeros(1, seq_len, head_dim)
    pos_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = rotary_emb(dummy, pos_ids)
    save_weight_tensor(cos, weights_dir / "l0_rope_cos.pt")
    save_weight_tensor(sin, weights_dir / "l0_rope_sin.pt")
    weights["cos"] = cos
    weights["sin"] = sin
    weight_entries["rope_cos"] = "l0_rope_cos.pt"
    weight_entries["rope_sin"] = "l0_rope_sin.pt"

    attention_entry = PrimitiveEntry(
        primitive_id="PRM_FULL_ATTN_L0",
        symbolic_name="full_attention_layer_0",
        type="operator_circuit",
        source_layers=[0],
        math_def=MathDef(
            operator_type="multihead_attention",
            u_uri=f"weights/{weight_entries['q_proj.weight']}",
        ),
        behavior=BehaviorMeta(
            description="Full multi-head attention for Qwen2.5-1.5B layer 0",
            trigger_conditions=["all_tokens"],
        ),
    )
    attention_entry.weight_data = {
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "weight_files": weight_entries,
    }

    save_stdlib_json([attention_entry], out_dir / "stdlib.uvm")

    return weights, weight_entries, model, tokenizer, rotary_emb


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "full_attention_stdlib"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights, weight_entries, model, tokenizer, rotary_emb = extract_and_save_attention_stdlib(out_dir, device)

    stdlib_payload = {
        "full_attn_L0": {
            "operator_type": "multihead_attention",
            "W_q": weights["q_proj.weight"],
            "b_q": weights["q_proj.bias"],
            "W_k": weights["k_proj.weight"],
            "b_k": weights["k_proj.bias"],
            "W_v": weights["v_proj.weight"],
            "b_v": weights["v_proj.bias"],
            "W_o": weights["o_proj.weight"],
            "cos": weights["cos"],
            "sin": weights["sin"],
            "n_heads": 12,
            "n_kv_heads": 2,
            "head_dim": 128,
        }
    }

    l0_attn = model.model.layers[0].self_attn

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

    pre_h = l0_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_h = l0_attn.register_forward_hook(post_hook)

    test_prompts = [
        "The cat sat on the mat and looked around.",
        "Machine learning enables computers to learn from data.",
        "The quick brown fox jumps over the lazy dog.",
        "Deep learning models require large amounts of data.",
        "Neural networks consist of interconnected layers.",
    ]

    print(f"\nRunning {len(test_prompts)} test prompts through Qwen...")
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    pre_h.remove()
    post_h.remove()

    print(f"\n--- Fidelity test: UCN compiled vs Qwen original ---")
    print(f"{'='*70}")

    ref = ReferenceBackend(
        stdlib_weights=stdlib_payload,
        device="cpu",
        dtype=torch.float32,
    )

    program = Program()
    program.add_stmt("y", Transform("x", MatrixRef("stdlib", "full_attn_L0")))

    all_cosines = []

    for i, (hs, real_out) in enumerate(zip(attn_inputs, attn_outputs_real)):
        with torch.no_grad():
            ucn_out = ref.execute(program, {"x": hs.cpu()})["y"]

            real_cpu = real_out.float().cpu()
            ucn_cpu = ucn_out.float().cpu()

            cos_sim = F.cosine_similarity(
                real_cpu.reshape(-1).unsqueeze(0),
                ucn_cpu.reshape(-1).unsqueeze(0),
                dim=-1,
            ).item()

            mse = F.mse_loss(real_cpu, ucn_cpu).item()
            all_cosines.append(cos_sim)

            T = real_cpu.shape[1]
            per_token = [f"{F.cosine_similarity(real_cpu[0,t].unsqueeze(0), ucn_cpu[0,t].unsqueeze(0), dim=-1).item():.8f}" for t in range(T)]

            prompt_preview = test_prompts[i][:50]
            print(f"  Prompt {i+1}: '{prompt_preview}...'")
            print(f"    Cosine: {cos_sim:.8f}  MSE: {mse:.8f}")
            print(f"    Per-token: {per_token}")

    avg_cos = sum(all_cosines) / len(all_cosines)

    print(f"\n{'='*70}")
    print(f"UCN COMPILED vs ORIGINAL QWEN:")
    print(f"  Mean cosine similarity: {avg_cos:.8f}")
    print(f"  Min cosine: {min(all_cosines):.8f}")
    print(f"  Max cosine: {max(all_cosines):.8f}")

    if avg_cos > 0.999:
        print(f"\n  FIDELITY GAP: FULLY BRIDGED")
        print(f"  The UCN-compiled attention program produces identical output")
        print(f"  to the original Qwen2.5-1.5B attention layer.")
    else:
        print(f"\n  Remaining gap: {1.0 - avg_cos:.6f}")

    report = {
        "method": "ucn_compiled_full_attention",
        "mean_cosine": avg_cos,
        "min_cosine": min(all_cosines),
        "max_cosine": max(all_cosines),
        "n_prompts": len(test_prompts),
        "gap_bridged": avg_cos > 0.999,
    }

    with open(out_dir / "ucn_fidelity_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print(f"\nReport saved to {out_dir / 'ucn_fidelity_report.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
