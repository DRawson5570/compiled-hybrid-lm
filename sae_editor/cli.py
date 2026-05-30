from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from sae_editor.pipeline import NRTCSPipeline
from sae_editor.recompiler import (
    RecompilerEngine,
    build_dense_map,
    orthogonal_projection,
)
from sae_editor.splicer import SafetensorsSplicer


def cmd_decompile(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    d_model = model.config.hidden_size

    if args.sae_path:
        from sae_editor.sae_training import SAERegistry
        saes = SAERegistry.load(args.sae_path, d_model=d_model, n_features=args.n_features)
    else:
        from sae_editor.tests.utils import make_random_sae
        print("No SAE path provided, using random (untrained) SAEs.")
        saes = {l: make_random_sae(d_model=d_model, n_features=args.n_features)
                for l in range(model.config.num_hidden_layers)}

    from sae_editor.decompiler import NRTCSDecompiler
    decompiler = NRTCSDecompiler(
        model=model, tokenizer=tokenizer, saes=saes,
        threshold=args.threshold, device=str(next(model.parameters()).device),
    )

    texts = args.text or ["The capital of France is Paris."]
    features = decompiler.extract_features(texts, max_length=args.max_length)

    torch.save(features, args.output)
    for layer_idx, fdata in features.items():
        n_f = len(fdata["feature_indices"])
        print(f"  Layer {layer_idx}: {n_f} active features")
    print(f"Saved features to {args.output}")


def cmd_recompile(args):
    engine = RecompilerEngine(eps=args.eps)
    edits = _load_edits(args.edits_path)

    original_features = None
    if args.features_path:
        original_features = _load_features(args.features_path)

    patches = {}
    for layer_idx, edit in edits.items():
        feats = None
        if original_features is not None:
            feats = original_features.get(layer_idx)
        patches[layer_idx] = engine.compile(
            edit["keys"], edit["values"],
            feats,
        )

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    for layer_idx, patch in patches.items():
        layer_dir = output_path / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)
        torch.save(patch["W_down"], layer_dir / "W_down.pt")
        torch.save(patch["W_up"], layer_dir / "W_up.pt")

    first_layer = next(iter(edits))
    verification = engine.verify(
        edits[first_layer]["keys"],
        patches[first_layer]["W_down"],
        patches[first_layer]["W_up"],
    )
    mean_err = (verification - edits[first_layer]["values"]).norm(dim=-1).mean().item()
    print(f"Compiled {len(patches)} layer patches. Layer {first_layer} mean reconstruction error: {mean_err:.6f}")


def cmd_splice(args):
    with SafetensorsSplicer(args.safetensors_path) as spl:
        names = spl.tensor_names
        matching = [n for n in names if args.tensor_name in n]
        print(f"Tensor names matching '{args.tensor_name}':")
        for n in sorted(matching):
            info = spl.get_tensor_info(n)
            print(f"  {n}: shape={info['shape']}, dtype={info['dtype']}")

    if args.patch_path:
        patches = _load_patches(args.patch_path)
        with SafetensorsSplicer(args.safetensors_path) as spl:
            for layer_idx, patch in patches.items():
                W_down = patch["W_down"]
                W_up = patch["W_up"]
                spl.splice_mlp(layer=layer_idx, W_down=W_down, W_up=W_up)
                print(f"Spliced layer {layer_idx}")


def cmd_roundtrip(args):
    pipeline = NRTCSPipeline(eps=args.eps)
    edits = _load_edits(args.edits_path)
    original_features = None
    if args.features_path:
        original_features = _load_features(args.features_path)

    patches = pipeline.round_trip(
        args.safetensors_path,
        edits,
        original_features,
        model_prefix=args.model_prefix,
    )

    verification = pipeline.verify_compilation(edits, patches)
    for layer_idx, v in verification.items():
        print(
            f"Layer {layer_idx}: "
            f"max_err={v['max_error']:.6f} "
            f"mean_err={v['mean_error']:.6f} "
            f"mean_cos={v['mean_cosine']:.6f}"
        )
    print(f"Patched {len(edits)} layers in {args.safetensors_path}")


def cmd_preview(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pipeline = NRTCSPipeline(eps=args.eps)
    edits = _load_edits(args.edits_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32,
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()

    prompt = args.prompt or "The capital of France is"

    if args.compare:
        strengths = [float(s) for s in args.compare.split(",")]
        results = pipeline.compare(edits, model, tokenizer, [prompt], strengths=strengths)
        for i, result in enumerate(results):
            print(f"\nStrength={strengths[i]:.1f}: cosine_shift={result.combined_cosine_shift:.4f}, "
                  f"combined_top1={result.combined_top_k[0] if result.combined_top_k else 'N/A'}")
    else:
        result = pipeline.preview(
            edits, model, tokenizer, [prompt],
            strength=args.strength,
            max_new_tokens=args.generate,
        )
        print(f"Preview: strength={args.strength}, "
              f"combined_cosine={result.combined_cosine_shift:.4f}")
        print(f"  Original top-3: {result.combined_top_k[:3]}")
        for layer_idx, r in result.per_layer.items():
            print(f"  Layer {layer_idx}: recon_err={r.reconstruction_error:.6f}, "
                  f"offset_l2={r.offset_l2:.4f}")
        if args.generate > 0:
            print(f"  Original: {result.original_text[:200]}")
            print(f"  Patched:  {result.patched_text[:200]}")


def cmd_library(args):
    from sae_editor.kv_library import KVLibrary

    lib = KVLibrary(args.path)

    if args.action == "list":
        ids = lib.list()
        for eid in ids:
            entry = lib.get(eid)
            print(f"  {eid}: {entry.description} [tags: {', '.join(entry.tags)}]")
        if not ids:
            print("  Library is empty.")

    elif args.action == "search":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
        results = lib.search(query=args.query or "", tags=tags)
        for entry in results:
            print(f"  {entry.entry_id}: {entry.description}")
            print(f"    layer={entry.layer}, tags={entry.tags}, "
                  f"verif_cos={entry.verification_cosine}")
        if not results:
            print("  No matches.")

    elif args.action == "preview":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True,
            torch_dtype=torch.float16, attn_implementation="eager",
        ).cuda()
        model.eval()

        entry_ids = [e.strip() for e in args.entry_ids.split(",") if e.strip()]
        prompt = args.prompt or "Test prompt"
        result = lib.preview(entry_ids, model, tokenizer, [prompt],
                             strength=args.strength, gate_threshold=args.gate_threshold)
        print(f"  Cosine shift: {result.combined_cosine_shift:.4f}")
        for lidx, r in result.per_layer.items():
            print(f"  Layer {lidx}: recon_err={r.reconstruction_error:.6f}, "
                  f"offset_l2={r.offset_l2:.4f}")
        del model
        torch.cuda.empty_cache()

    elif args.action == "splice":
        entry_ids = [e.strip() for e in args.entry_ids.split(",") if e.strip()]
        lib.splice(entry_ids, args.safetensors)
        print(f"  Spliced {entry_ids} into {args.safetensors}")


def cmd_parse(args):
    from sae_editor.dsl.nrtcs_parser import parse_nrtcs, serialize_nrtcs

    if args.direction == "parse":
        with open(args.input) as f:
            source = f.read()
        edits = parse_nrtcs(source)
        torch.save(edits, args.output)
        print(f"Parsed {len(edits)} layers to {args.output}")
    elif args.direction == "serialize":
        edits = _load_edits(args.input)
        output = serialize_nrtcs(edits)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Serialized to {args.output}")
        else:
            print(output)


def _load_edits(path: str) -> dict:
    data = torch.load(path, weights_only=False)
    if isinstance(data, dict) and "keys" in data and "values" in data:
        return {0: data}
    return data


def _load_features(path: str) -> dict:
    data = torch.load(path, weights_only=False)
    if isinstance(data, torch.Tensor):
        return {0: data}
    return data


def _load_patches(path: str) -> dict:
    patch_dir = Path(path)
    patches = {}
    for layer_dir in sorted(patch_dir.iterdir()):
        if layer_dir.is_dir() and layer_dir.name.startswith("layer_"):
            layer_idx = int(layer_dir.name.split("_")[1])
            W_down = torch.load(layer_dir / "W_down.pt", weights_only=False)
            W_up = torch.load(layer_dir / "W_up.pt", weights_only=False)
            patches[layer_idx] = {"W_down": W_down, "W_up": W_up}
    return patches


def main():
    parser = argparse.ArgumentParser(description="NRTCS CLI")
    sub = parser.add_subparsers(dest="command")

    p_decompile = sub.add_parser("decompile", help="Phase 1: decompile model")
    p_decompile.add_argument("model_path")
    p_decompile.add_argument("--sae-path", default=None, help="Path to trained SAE directory")
    p_decompile.add_argument("--n-features", type=int, default=64)
    p_decompile.add_argument("--text", nargs="*", default=None)
    p_decompile.add_argument("--max-length", type=int, default=128)
    p_decompile.add_argument("--output", default="decompiled.pt")
    p_decompile.add_argument("--threshold", type=float, default=0.1)

    p_recompile = sub.add_parser("recompile", help="Phase 3: compile to weights")
    p_recompile.add_argument("edits_path")
    p_recompile.add_argument("--features-path", default=None)
    p_recompile.add_argument("--output", default="patches/")
    p_recompile.add_argument("--eps", type=float, default=1e-6)

    p_splice = sub.add_parser("splice", help="Phase 4: inspect/patch safetensors")
    p_splice.add_argument("safetensors_path")
    p_splice.add_argument("--tensor-name", default="mlp")
    p_splice.add_argument("--patch-path", default=None)

    p_rt = sub.add_parser("roundtrip", help="Full round-trip compile + splice")
    p_rt.add_argument("edits_path")
    p_rt.add_argument("safetensors_path")
    p_rt.add_argument("--features-path", default=None)
    p_rt.add_argument("--model-prefix", default="model.layers.{layer}.mlp")
    p_rt.add_argument("--eps", type=float, default=1e-6)

    p_parse = sub.add_parser("parse", help="Parse/serialize NRTCS DSL")
    p_parse.add_argument("direction", choices=["parse", "serialize"])
    p_parse.add_argument("input")
    p_parse.add_argument("--output", default=None)

    p_preview = sub.add_parser("preview", help="Preview patches without modifying model")
    p_preview.add_argument("edits_path")
    p_preview.add_argument("model_path")
    p_preview.add_argument("--prompt", default=None)
    p_preview.add_argument("--strength", type=float, default=1.0)
    p_preview.add_argument("--compare", default=None, help="Comma-separated strengths to sweep")
    p_preview.add_argument("--generate", type=int, default=0, help="Max new tokens for generation")
    p_preview.add_argument("--eps", type=float, default=1e-6)
    p_preview.add_argument("--gate-threshold", type=float, default=0.3,
                           help="Cosine similarity threshold for gated injection")

    p_lib = sub.add_parser("library", help="Manage key-value patch library")
    p_lib.add_argument("--path", default="patches/qwen2.5-0.5b/",
                       help="Path to library directory")
    lib_sub = p_lib.add_subparsers(dest="action")

    p_lib_list = lib_sub.add_parser("list", help="List all entries")

    p_lib_search = lib_sub.add_parser("search", help="Search entries")
    p_lib_search.add_argument("--query", default="")
    p_lib_search.add_argument("--tags", default=None, help="Comma-separated tags")

    p_lib_preview = lib_sub.add_parser("preview", help="Preview entries")
    p_lib_preview.add_argument("entry_ids", help="Comma-separated entry IDs")
    p_lib_preview.add_argument("--model", required=True)
    p_lib_preview.add_argument("--prompt", default=None)
    p_lib_preview.add_argument("--strength", type=float, default=1.0)
    p_lib_preview.add_argument("--gate-threshold", type=float, default=0.3)

    p_lib_splice = lib_sub.add_parser("splice", help="Splice entries into model")
    p_lib_splice.add_argument("entry_ids", help="Comma-separated entry IDs")
    p_lib_splice.add_argument("--safetensors", required=True)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    handlers = {
        "decompile": cmd_decompile,
        "recompile": cmd_recompile,
        "splice": cmd_splice,
        "roundtrip": cmd_roundtrip,
        "parse": cmd_parse,
        "preview": cmd_preview,
        "library": cmd_library,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
