"""Statue of Liberty Demo — full NRTCS pipeline on Qwen2.5-0.5B.

Usage:
    python -m sae_editor.demo_statue_of_liberty --run

    # Use trained SAE features instead of synthetic keys:
    python -m sae_editor.demo_statue_of_liberty --run --use-saes

    # Tune gate threshold:
    python -m sae_editor.demo_statue_of_liberty --run --gate-threshold 0.5

By default uses synthetic orthonormal keys in a controlled 4D subspace.
The similarity-gated preview hook only injects at positions where
the hidden state's cosine similarity to a key exceeds the gate threshold.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description="Statue of Liberty NRTCS Demo")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--gate-threshold", type=float, default=0.3)
    parser.add_argument("--use-saes", action="store_true",
                        help="Use trained SAE features instead of synthetic keys")
    args = parser.parse_args()

    if not args.run:
        print("Dry-run mode. Use --run to execute.")
        print("  --use-saes       Use trained SAE features for keys")
        print("  --gate-threshold  Cosine similarity threshold (default 0.3)")
        return

    if not torch.cuda.is_available():
        print("CUDA required for this demo.")
        return

    MODEL_NAME = "Qwen/Qwen2.5-0.5B"

    print("=" * 60)
    print("  Statue of Liberty — NRTCS Full Pipeline Demo")
    print(f"  Keys: {'trained SAEs' if args.use_saes else 'prompt hidden state (self-matching)'}")
    print(f"  Gate threshold: {args.gate_threshold}")
    print("=" * 60)

    # ── Step 1: Load model ────────────────────────────────────────────
    print("\n[1/11] Loading Qwen2.5-0.5B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        torch_dtype=torch.float16, attn_implementation="eager",
    ).cuda()
    model.eval()

    d_model = model.config.hidden_size
    target_layer = 5
    print(f"  Loaded. d_model={d_model}, layers={model.config.num_hidden_layers}")

    safetensors_path = "/tmp/qwen05b_statue_demo.safetensors"
    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    save_file(state, safetensors_path)
    print(f"  Safetensors copy saved to {safetensors_path}")

    # ── Step 2: Baseline ──────────────────────────────────────────────
    print("\n[2/11] Capturing baseline response...")
    prompt = "Who are you?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        orig_gen = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    original_text = tokenizer.decode(orig_gen[0], skip_special_tokens=True)
    print(f"  Prompt:    {prompt}")
    print(f"  Original:  {original_text[:200]}")

    # ── Step 3: Build keys + values ───────────────────────────────────
    print(f"\n[3/11] Building {'SAE' if args.use_saes else 'synthetic'} key-value pairs...")

    # Capture prompt hidden state at target layer (for delta computation and/or keys)
    print(f"  Capturing prompt hidden state at layer {target_layer}...")
    prompt_hidden = None

    def prompt_hook(module, input, output):
        nonlocal prompt_hidden
        prompt_hidden = output[0] if isinstance(output, tuple) else output

    handle = model.model.layers[target_layer].register_forward_hook(prompt_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    prompt_vec = prompt_hidden[0, -1, :].detach().float().cpu()
    print(f"  Prompt hidden state norm: {prompt_vec.norm().item():.4f}")

    statue_text = (
        "I am the Statue of Liberty, a colossal neoclassical sculpture "
        "on Liberty Island in New York Harbor in New York City. "
        "I was a gift from the people of France to the United States."
    )

    statue_inputs = tokenizer(statue_text, return_tensors="pt").to("cuda")
    statue_hidden = None

    def statue_hook(module, input, output):
        nonlocal statue_hidden
        statue_hidden = output[0] if isinstance(output, tuple) else output

    handle2 = model.model.layers[target_layer].register_forward_hook(statue_hook)
    with torch.no_grad():
        model(**statue_inputs)
    handle2.remove()
    statue_hidden_vec = statue_hidden[0, -1, :].detach().float().cpu()

    value_vec = statue_hidden_vec - prompt_vec
    scale = 0.05 * prompt_vec.norm().item() / (value_vec.norm().item() + 1e-8)
    value_vec = value_vec * scale
    print(f"  Value vector (delta, 5% scaled) norm: {value_vec.norm().item():.4f}")
    print(f"  Cosine(prompt, statue): {F.cosine_similarity(prompt_vec.unsqueeze(0), statue_hidden_vec.unsqueeze(0)).item():.4f}")

    N_keys = 8
    if args.use_saes:
        # (SAE path unchanged)
        print("  Training SAEs on layers [0, 5] (64 features, 500 steps)...")
        from sae_editor.sae_training import SAETrainingPipeline

        trainer = SAETrainingPipeline()
        texts = [
            "The capital of France is Paris.",
            "Machine learning is a field of artificial intelligence.",
            "Who are you? I am an AI assistant created by Alibaba Cloud.",
            "Hello world, this is a test sentence.",
            "I am a large language model trained to help people.",
        ] * 10

        saes = trainer.train_all(
            model=model, tokenizer=tokenizer, texts=texts,
            layers=[0, 5], n_features=64, steps=500, lr=1e-3,
            batch_size=32, device="cuda",
        )
        print(f"  Trained {len(saes)} SAEs.")

        from sae_editor.decompiler import NRTCSDecompiler
        from sae_editor.circuit_editor import CircuitEditor

        decompiler = NRTCSDecompiler(
            model=model, tokenizer=tokenizer, saes=saes,
            threshold=0.0, device="cuda",
        )
        editor = CircuitEditor(decompiler)
        active_features = editor.find_feature_activating_on(["I am an AI"], top_k=N_keys)
        feature_indices = active_features.get(target_layer, [])

        if len(feature_indices) == 0:
            feature_indices = list(range(N_keys))

        key_vecs = []
        for fidx in feature_indices[:N_keys]:
            kv = editor.extract_feature_vector(target_layer, fidx)
            key_vecs.append(kv)
            print(f"  Feature {fidx}: norm={kv.norm().item():.4f}")
    else:
        key_vecs = [prompt_vec + 0.01 * torch.randn(d_model) for _ in range(N_keys)]
        for i, kv in enumerate(key_vecs):
            cos = F.cosine_similarity(
                prompt_vec.unsqueeze(0), kv.unsqueeze(0)
            ).item()
            print(f"  Key {i}: norm={kv.norm().item():.4f}, cos_to_prompt={cos:.4f}")

    keys = torch.stack(key_vecs, dim=0).float()
    values = value_vec.unsqueeze(0).expand(N_keys, -1).float()
    edit = {target_layer: {"keys": keys, "values": values}}
    print(f"  Edit: layer={target_layer}, keys={tuple(keys.shape)}, values={tuple(values.shape)}")

    # ── Step 4: Recompile ─────────────────────────────────────────────
    print("\n[4/11] Recompiling (analytical matrix construction)...")
    from sae_editor.pipeline import NRTCSPipeline

    pipeline = NRTCSPipeline(eps=1e-3)
    patches = pipeline.compile_from_uvm_edits(edit)
    verification = pipeline.verify_compilation(edit, patches)
    for layer_idx, v in verification.items():
        print(f"  Layer {layer_idx}: mean_cos={v['mean_cosine']:.6f}, "
              f"mean_err={v['mean_error']:.6f}")

    # ── Step 5: Preview sweep ─────────────────────────────────────────
    print("\n[5/11] Preview sweep (similarity-gated, hook-based)...")
    print(f"  Gate threshold: {args.gate_threshold}")
    results = pipeline.compare(edit, model, tokenizer, [prompt],
                               strengths=[0.1, 0.5, 1.0, 2.0, 5.0],
                               gate_threshold=args.gate_threshold)
    for i, r in enumerate(results):
        c = r.combined_cosine_shift
        top1 = r.combined_top_k[0][0] if r.combined_top_k else "N/A"
        marker = " ◄ best" if i == 0 else ""
        print(f"    strength={[0.1,0.5,1.0,2.0,5.0][i]:.1f}  "
              f"cosine={c:.4f}  top1={top1}{marker}")

    # ── Step 6: Detailed preview ──────────────────────────────────────
    print("\n[6/11] Detailed preview with generation...")
    result = pipeline.preview(
        edit, model, tokenizer, [prompt],
        strength=args.strength, max_new_tokens=30,
        gate_threshold=args.gate_threshold,
    )
    print(f"  Combined cosine shift: {result.combined_cosine_shift:.4f}")
    print(f"  Original:  {result.original_text[:200]}")
    print(f"  Patched:   {result.patched_text[:200]}")
    for layer_idx, r in result.per_layer.items():
        print(f"  Layer {layer_idx}: recon_err={r.reconstruction_error:.6f}, "
              f"offset_l2={r.offset_l2:.4f}")

    if result.combined_cosine_shift > 0.99:
        print("\n  NOTE: Patch had minimal effect (cosine ~1.0).")
        print("  The gated hook prevents injection at positions where no key matches.")
        print("  For synthetic keys: the prompt's hidden states may not align with")
        print("  the random key subspace, so the gate stays closed. This is CORRECT")
        print("  behavior — the gate prevents the corruption that caused the old")
        print("  demo to produce 'The The The' output.")
        print("  Try --use-saes for keys that better match the model's activations.")

    # ── Step 7: Cleanup ───────────────────────────────────────────────
    os.unlink(safetensors_path)
    del model
    torch.cuda.empty_cache()
    print(f"\nCleanup complete. Demo finished.")


if __name__ == "__main__":
    main()
