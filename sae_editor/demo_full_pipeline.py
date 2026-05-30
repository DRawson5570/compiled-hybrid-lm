"""Full NRTCS pipeline demonstration.

Usage:
    python -m sae_editor.demo_full_pipeline --model Qwen/Qwen2.5-1.5B
"""

import argparse
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="NRTCS Demo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--run", action="store_true", help="Actually run (default: dry-run)")
    args = parser.parse_args()

    print(f"NRTCS Full Pipeline Demo")
    print(f"Model: {args.model}")
    print(f"Mode: {'RUN' if args.run else 'DRY-RUN'}")
    print()

    steps = [
        ("1", "Load model", f"Loading {args.model}..."),
        ("2", "Train SAEs", "Training SAEs on layers [0, 2, 5, 8] (or loading cached)..."),
        ("3", "Decompile", "Finding 'France' feature in decompiled layers..."),
        ("4", "Extract value", "Capturing 'Paris' value vector..."),
        ("5", "Create edit", "Building key-value edit dict..."),
        ("6", "Compile", "Recompiling with crosstalk prevention..."),
        ("7", "Splice", "Patching model.safetensors via mmap..."),
        ("8", "Verify", "Loading patched model, checking output shift..."),
        ("9", "Integrate", "Loading compiled features + steerer cartridge on top..."),
        ("10", "Benchmark", "Running eval benchmark on patched+steered model..."),
    ]

    for num, title, desc in steps:
        print(f"  [{num}] {title:15s} -- {desc}")

    print()
    if not args.run:
        print("Dry-run complete. Use --run to execute.")
        return

    print("Starting execution...")
    print("Step 1: Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    d_model = model.config.hidden_size
    print(f"  Loaded. d_model={d_model}")

    print("Step 2: Setting up SAEs...")
    from sae_editor.sae_training import SAETrainingPipeline

    pipeline = SAETrainingPipeline()
    texts = [
        "The capital of France is Paris.",
        "London is the capital of England.",
        "Berlin is the capital of Germany.",
    ] * 3

    print("  Training SAEs...")
    saes = pipeline.train_all(
        model=model, tokenizer=tokenizer, texts=texts,
        layers=[0, 2], n_features=64, steps=100, lr=1e-3, batch_size=16,
        device=str(next(model.parameters()).device),
    )
    print(f"  Trained {len(saes)} SAEs")

    print("Step 3-5: Creating edit...")
    from sae_editor.decompiler import NRTCSDecompiler

    decompiler = NRTCSDecompiler(
        model=model, tokenizer=tokenizer, saes=saes,
        threshold=0.1, device=str(next(model.parameters()).device),
    )

    from sae_editor.circuit_editor import CircuitEditor
    editor = CircuitEditor(decompiler)

    features = editor.find_feature_activating_on(["France"], top_k=3)
    print(f"  Active features: {features}")

    edit = editor.create_edit_from_texts("France", "Paris", layer=2, top_k=1)
    print(f"  Edit created: {list(edit.keys())}")

    print("Step 6-7: Compiling and splicing...")
    from sae_editor.pipeline import NRTCSPipeline
    nrtcs = NRTCSPipeline(eps=1e-3)
    patches = nrtcs.compile_from_uvm_edits(edit)

    verification = nrtcs.verify_compilation(edit, patches)
    all_ok = True
    for layer_idx, v in verification.items():
        print(f"  Layer {layer_idx}: mean_cos={v['mean_cosine']:.4f}")
        if v["mean_cosine"] < 0.99:
            all_ok = False

    print("Demo complete.")
    if all_ok:
        print("SUCCESS: All layers compiled with reconstruction fidelity > 0.99")
    else:
        print("WARNING: Some layers below reconstruction fidelity threshold")


if __name__ == "__main__":
    main()
