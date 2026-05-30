"""
Phase 3: Verify that an extracted Qwen2.5-1.5B attention head can be compiled
to a UCN program and produces faithful output.

Measures:
1. Cosine similarity between UCN output and original attention head output
2. MSE of the attention output reconstruction
3. Per-token fidelity

The approach: hook into the self_attn submodule, capture input and output,
compute the equivalent UCN program, and compare.
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


def extract_copy_head_weights():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )

    l0_attn = model.model.layers[0].self_attn

    v_head = l0_attn.v_proj.weight.data[128:256, :].clone().float()
    o_head = l0_attn.o_proj.weight.data[:, 1024:1152].clone().float()

    return v_head, o_head, model


def build_copy_head_stdlib(v_head, o_head, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    u = v_head.clone()
    v = o_head.T.clone()

    save_weight_tensor(u, weights_dir / "copy_head_v_u.pt")
    save_weight_tensor(v, weights_dir / "copy_head_o_v.pt")

    entry = PrimitiveEntry(
        primitive_id="PRM_COPY_HEAD_L0_H8",
        symbolic_name="copy_head_L0_H8",
        type="operator_circuit",
        source_layers=[0],
        math_def=MathDef(
            operator_type="low_rank_projection",
            rank=128,
            u_uri="weights/copy_head_v_u.pt",
            v_uri="weights/copy_head_o_v.pt",
        ),
        behavior=BehaviorMeta(
            description="Copy head V/O projection: attends to previous token, copies via V/O",
            trigger_conditions=["sequence_tokens", "previous_position"],
        ),
    )

    save_stdlib_json([entry], out_dir / "stdlib.uvm")

    return {
        "u": u,
        "v": v,
    }


def run_fidelity_test(device="cuda"):
    out_dir = Path(__file__).resolve().parent.parent / "artifacts" / "copy_head_fidelity"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting copy head weights from Qwen2.5-1.5B...")
    v_head, o_head, model = extract_copy_head_weights()
    model.eval()
    model.to(device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

    print("Building stdlib...")
    weights = build_copy_head_stdlib(v_head, o_head, out_dir)

    stdlib = {
        "copy_head_L0_H8": {
            "operator_type": "low_rank_projection",
            "u": weights["u"],
            "v": weights["v"],
        }
    }

    test_prompts = [
        "The cat sat on the mat and looked around.",
        "Machine learning enables computers to learn from data.",
        "The capital of France is Paris, a city known for its art.",
        "Python is a high-level programming language for data science.",
        "The quick brown fox jumps over the lazy dog near the river.",
    ]

    print(f"\n{'='*70}")
    print(f"Running fidelity test on {len(test_prompts)} prompts")
    print(f"Comparing: UCN copy head output vs actual attention head output")
    print(f"{'='*70}")

    all_cosine = []
    all_mse = []
    l0_attn = model.model.layers[0].self_attn

    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"\n--- Prompt {prompt_idx+1}: '{prompt[:60]}...' ---")

        attn_inputs = []
        attn_outputs = []

        def pre_hook(module, args, kwargs):
            if 'hidden_states' in kwargs:
                attn_inputs.append(kwargs['hidden_states'].detach().clone())

        def post_hook(module, input, output):
            if isinstance(output, tuple):
                attn_outputs.append(output[0].detach().clone())
            else:
                attn_outputs.append(output.detach().clone())

        handle_pre = l0_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
        handle_post = l0_attn.register_forward_hook(post_hook)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

        handle_pre.remove()
        handle_post.remove()

        if not attn_inputs or not attn_outputs:
            print("  FAILED: No hook data collected")
            continue

        x_in = attn_inputs[0]
        h_attn_real = attn_outputs[0]

        shifted = torch.zeros_like(x_in)
        shifted[:, 0, :] = x_in[:, 0, :]
        if x_in.shape[1] > 1:
            shifted[:, 1:, :] = x_in[:, :-1, :]

        b, seq, d = x_in.shape
        x_flat = shifted.reshape(-1, d).float().cpu()

        ref = ReferenceBackend(stdlib_weights=stdlib, device="cpu", dtype=torch.float32)
        program = Program()
        program.add_stmt("y", Transform("x", MatrixRef("stdlib", "copy_head_L0_H8")))

        h_attn_ucn_flat = ref.execute(program, {"x": x_flat})["y"]
        h_attn_ucn = h_attn_ucn_flat.reshape(b, seq, d)

        h_attn_real_flat = h_attn_real.reshape(-1, d).float().cpu()

        cos_sim = F.cosine_similarity(
            h_attn_ucn.reshape(-1).unsqueeze(0),
            h_attn_real_flat.reshape(-1).unsqueeze(0),
            dim=-1,
        ).item()
        mse = F.mse_loss(h_attn_ucn, h_attn_real.float().cpu()).item()

        all_cosine.append(cos_sim)
        all_mse.append(mse)

        per_token_sims = []
        for t in range(seq):
            sim = F.cosine_similarity(
                h_attn_ucn[0, t].unsqueeze(0),
                h_attn_real[0, t].float().cpu().unsqueeze(0),
                dim=-1,
            ).item()
            per_token_sims.append(sim)

        norm_real = h_attn_real.float().norm(p=2, dim=-1)
        norm_ucn = h_attn_ucn.norm(p=2, dim=-1)

        printable_prompt = prompt[:40]
        print(f"  Overall cosine: {cos_sim:.6f}")
        print(f"  Overall MSE: {mse:.6f}")
        print(f"  Per-token cosine: {[f'{s:.4f}' for s in per_token_sims]}")
        print(f"  Norm real range: [{norm_real.min():.4f}, {norm_real.max():.4f}]")
        print(f"  Norm UCN range:  [{norm_ucn.min():.4f}, {norm_ucn.max():.4f}]")

    print(f"\n{'='*70}")
    print(f"Summary across {len(test_prompts)} prompts:")
    print(f"{'='*70}")

    avg_cos = sum(all_cosine) / len(all_cosine)
    avg_mse = sum(all_mse) / len(all_mse)
    std_cos = torch.tensor(all_cosine).std().item()
    std_mse = torch.tensor(all_mse).std().item()

    print(f"  Cosine similarity: {avg_cos:.6f} ± {std_cos:.6f}")
    print(f"  MSE: {avg_mse:.6f} ± {std_mse:.6f}")

    results = {
        "prompts": test_prompts,
        "cosine_similarities": all_cosine,
        "mses": all_mse,
        "avg_cosine": avg_cos,
        "avg_mse": avg_mse,
        "std_cosine": std_cos,
        "std_mse": std_mse,
    }

    with open(out_dir / "fidelity_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\nResults saved to {out_dir / 'fidelity_results.json'}")

    if avg_cos > 0.85:
        print("\nSUCCESS: Copy head compiled with >85% cosine similarity!")
    elif avg_cos > 0.50:
        print("\nPARTIAL: Copy head compiled but fidelity needs improvement.")
    else:
        print("\nLOW FIDELITY: The V*O projection alone doesn't capture the head's behavior. Need attention weights too.")

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_fidelity_test(device)
