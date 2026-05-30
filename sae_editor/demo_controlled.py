"""Controlled Minimal Demo — prove the NRTCS pipeline works.

A deterministic, fast demo that doesn't require SAE training.
Captures hidden states for two related prompts, computes a delta,
and demonstrates the gated preview sweep producing clean output.

Usage:
    python -m sae_editor.demo_controlled --run
    python -m sae_editor.demo_controlled --run --gate-threshold 0.5
"""

import argparse

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="NRTCS Controlled Minimal Demo")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gate-threshold", type=float, default=0.3)
    args = parser.parse_args()

    if not args.run:
        print("Dry-run. Use --run to execute.")
        return

    if not torch.cuda.is_available():
        print("CUDA required.")
        return

    MODEL = "Qwen/Qwen2.5-0.5B"
    LAYER = 5

    print("=" * 60)
    print("  Controlled NRTCS Demo — The sky is blue → red")
    print(f"  Gate threshold: {args.gate_threshold}")
    print("=" * 60)

    # Load
    print("\n[1/6] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True,
        torch_dtype=torch.float16, attn_implementation="eager",
    ).cuda()
    model.eval()
    d_model = model.config.hidden_size
    print(f"  Loaded. d_model={d_model}")

    # Capture hidden states
    print(f"\n[2/6] Capturing hidden states at layer {LAYER}...")

    def capture(text):
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        hidden = None

        def hook(module, input, output):
            nonlocal hidden
            hidden = (output[0] if isinstance(output, tuple) else output)[0, -1, :]

        handle = model.model.layers[LAYER].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        return hidden.detach().float().cpu()

    blue_hidden = capture("The sky is blue")
    red_hidden = capture("The sky is red")
    print(f"  Blue norm: {blue_hidden.norm().item():.4f}")
    print(f"  Red norm: {red_hidden.norm().item():.4f}")
    print(f"  Cosine(blue, red): {F.cosine_similarity(blue_hidden.unsqueeze(0), red_hidden.unsqueeze(0)).item():.4f}")

    # Build keys (blue prompt's state) and values (delta from blue to red)
    raw_delta = red_hidden - blue_hidden
    scale = 0.05 * blue_hidden.norm().item() / (raw_delta.norm().item() + 1e-8)
    delta = raw_delta * scale
    print(f"  Delta norm (5% scaled): {delta.norm().item():.4f}")

    N = 4
    keys = blue_hidden.unsqueeze(0) + 0.01 * torch.randn(N, d_model)
    values = delta.unsqueeze(0).expand(N, -1)
    edit = {LAYER: {"keys": keys, "values": values}}

    # Recompile
    print("\n[3/6] Recompiling...")
    from sae_editor.pipeline import NRTCSPipeline

    pipeline = NRTCSPipeline(eps=1e-3)
    patches = pipeline.compile_from_uvm_edits(edit)
    for layer_idx, v in pipeline.verify_compilation(edit, patches).items():
        print(f"  Layer {layer_idx}: mean_cos={v['mean_cosine']:.6f}, mean_err={v['mean_error']:.6f}")

    # Preview sweep
    print(f"\n[4/6] Preview sweep (gate_threshold={args.gate_threshold}):")
    results = pipeline.compare(
        edit, model, tokenizer, ["The sky is blue"],
        strengths=[0.1, 0.5, 1.0, 2.0, 3.0],
        gate_threshold=args.gate_threshold,
    )
    for i, r in enumerate(results):
        c = r.combined_cosine_shift
        top1 = r.combined_top_k[0][0] if r.combined_top_k else "N/A"
        top2 = r.combined_top_k[1][0] if len(r.combined_top_k) > 1 else "N/A"
        print(f"    s={[0.1,0.5,1.0,2.0,3.0][i]:.1f}  "
              f"cosine={c:.4f}  top1={top1}  top2={top2}")

    # Detailed preview
    print("\n[5/6] Detailed preview with generation...")
    result = pipeline.preview(
        edit, model, tokenizer, ["The sky is blue"],
        strength=1.0, max_new_tokens=20,
        gate_threshold=args.gate_threshold,
    )
    print(f"  Cosine shift: {result.combined_cosine_shift:.4f}")
    print(f"  Original: {result.original_text[:150]}")
    print(f"  Patched:  {result.patched_text[:150]}")
    for layer_idx, r in result.per_layer.items():
        print(f"  Layer {layer_idx}: recon_err={r.reconstruction_error:.6f}, offset_l2={r.offset_l2:.4f}")

    # Done
    del model
    torch.cuda.empty_cache()
    print("\n[6/6] Cleanup complete.")
    print("\n" + "=" * 60)
    if result.combined_cosine_shift < 0.99:
        print("  Pipeline confirmed: gated injection shifts output")
    else:
        print("  Gate stayed closed — prompt may not match key space")
    print("=" * 60)


if __name__ == "__main__":
    main()
