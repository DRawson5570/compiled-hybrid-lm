"""chat_cmi_v8000.py

Interactive terminal-based chat client demonstrating sequence-aware CMI (Capability Mixture of Experts)
next-token routing autoregression at V=8000 BPE scale.

Loads 5 compiled expert channels (PPMI, Bigram, Shape, Freq, Recency), loads the trained SimpleClassifier router,
and lets you type natural English prompts to generate continuation streams with dynamic expert blending.
"""
from __future__ import annotations

import math
import os
import sys
import time
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO = Path("/home/drawson/llm_decoupling")
sys.path.insert(0, str(REPO))
DS = Path("/home/drawson/deepseek_experiments")

from compile_wiki_lm_v13 import load_setup, load_or_build_tokens, DEVICE
from train_cmi_v8000 import build_all_channels, SimpleClassifier
from train_blender_v8000 import build_vectorized_features
from hybrid.v3_super_blender.model import WindowMLPBlender, LookbackMLPBlender, GRUBlender, CausalConvBlender

def format_weights_bar(pct: float, width: int = 15) -> str:
    """Returns a visual progress bar representing the percentage."""
    filled = int(round(pct * width))
    return f"[{'#' * filled}{' ' * (width - filled)}]"

def generate_step(
    ids_t: torch.Tensor,
    channels_lp: list[torch.Tensor],
    active_router_name: str,
    models: dict[str, nn.Module],
    emb: torch.Tensor,
    V: int,
    decoding_mode: str,
    temp_routing: float,
    temp_gen: float,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Performs a single next-token autoregressive prediction step.
    Returns (next_token_id, routing_probs, blended_probs)
    """
    C = len(channels_lp)
    
    # Prepend padding dummy token if sequence is of length 1 to avoid lag-1 shape errors
    if len(ids_t) == 1:
        ids_input = torch.cat([torch.tensor([0], device=ids_t.device), ids_t])
    else:
        ids_input = ids_t

    # 1. Feature extraction over the causal prefix sequence
    with torch.no_grad():
        features, _, _ = build_vectorized_features(ids_input, channels_lp, emb, ids_t.device)

    # 2. Get routing weights from selected router model
    with torch.no_grad():
        if active_router_name == "classifier":
            # Simple classifier expects (1, F_dim) of the last step
            feat_vec = features[-1].unsqueeze(0)
            router_logits = models["classifier"](feat_vec).squeeze(0)  # (5,)
            routing_probs = F.softmax(router_logits / max(temp_routing, 1e-5), dim=0)
        else:
            model = models[active_router_name]
            if active_router_name in ["window_mlp", "lookback_mlp"]:
                log_w = model(features, is_already_windowed=False)
            elif active_router_name == "gru":
                log_w, _ = model(features.unsqueeze(0))
                log_w = log_w.squeeze(0)
            elif active_router_name == "causal_conv":
                log_w = model(features.unsqueeze(0)).squeeze(0)
            
            # The weights for the last step (prediction of next token)
            routing_probs = log_w[-1].exp()

    # 3. Apply decoding mode rules
    # Mode can be "soft" (weighted blend), "hard" (argmax channel), or pure index
    blended_probs = torch.zeros(V, device=ids_t.device)
    x_o = ids_t[-1].item()

    if decoding_mode == "soft":
        for ch in range(C):
            blended_probs += routing_probs[ch] * channels_lp[ch][x_o].exp()
    elif decoding_mode == "hard":
        best_ch = routing_probs.argmax().item()
        blended_probs = channels_lp[best_ch][x_o].exp()
    elif decoding_mode.isdigit():
        ch_idx = int(decoding_mode)
        blended_probs = channels_lp[ch_idx][x_o].exp()
    else:
        # Fallback to soft
        for ch in range(C):
            blended_probs += routing_probs[ch] * channels_lp[ch][x_o].exp()

    # Normalize blended probabilities with robust fallback to avoid device-side assert failures
    sum_probs = blended_probs.sum()
    if sum_probs <= 1e-15 or torch.isnan(sum_probs) or torch.isinf(sum_probs):
        blended_probs = torch.ones_like(blended_probs) / V
    else:
        blended_probs = blended_probs / sum_probs

    # Ensure no NaN or Inf remains in blended_probs
    blended_probs = torch.nan_to_num(blended_probs, nan=1.0/V, posinf=1.0/V, neginf=0.0)
    sum_probs = blended_probs.sum()
    if sum_probs > 0:
        blended_probs = blended_probs / sum_probs
    else:
        blended_probs = torch.ones_like(blended_probs) / V

    # 4. Decoding and Sampling
    if temp_gen <= 1e-5:
        # Argmax decoding
        next_id = blended_probs.argmax().item()
    else:
        # Sample with temperature
        # Convert to logits space
        logits = torch.log(blended_probs.clamp(min=1e-30)) / temp_gen
        probs = F.softmax(logits, dim=0)
        # Ensure probs has no nans or infs
        probs = torch.nan_to_num(probs, nan=1.0/V)
        sum_p = probs.sum()
        if sum_p > 0:
            probs = probs / sum_p
        else:
            probs = torch.ones_like(probs) / V
        next_id = torch.multinomial(probs, 1).item()

    return next_id, routing_probs, blended_probs

def main():
    print("=" * 80)
    print("      CMI CAPABILITY FUSED EXPERT INTERACTIVE V=8000 CHAT CLIENT")
    print("=" * 80)

    # Setup the BPE tokenizer and embeddings
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = emb.to(target_device)
    print(f"Loaded BPE vocabulary: {V} tokens, Embedding size: {d} features.")

    # Load IDs from corpus to compute unigram/bigram stats
    print("Loading corpus tokens...")
    ids_all = load_or_build_tokens(bpe, bpe_to_lm, V)
    ids_np = ids_all.numpy()

    # Build the 5 CMI channels
    print("Compiling 5 corporate next-token channels on corpus statistics...")
    channels_lp, ch_names = build_all_channels(bpe, ids_np, emb, V)
    channels_lp = [lp.to(target_device) for lp in channels_lp]
    print("Done compiling channels.")

    # Load routing classifier
    classifier = SimpleClassifier(in_dim=276, n_classes=5).to(target_device)
    classifier_path = DS / "artifacts/cmi_v8000_classifier.pt"
    if classifier_path.exists():
        checkpoint = torch.load(classifier_path, map_location=target_device)
        classifier.load_state_dict(checkpoint["state_dict"])
        classifier.eval()
        print(f"Loaded SimpleClassifier router model (acc: {checkpoint.get('acc', 0.0):.2f})")
    else:
        print(f"Warning: {classifier_path} not found! Using untrained random weights.")

    models = {
        "classifier": classifier
    }

    # Initialize and load sequence-aware blenders
    F_dim = 276
    C_dim = 5
    blenders_dir = DS / "artifacts/compiled_wiki_lm_v8000_blenders"
    
    blenders_init = {
        "window_mlp": WindowMLPBlender(single_step_dim=F_dim, n_channels=C_dim, lookback_window=4, hidden=128),
        "lookback_mlp": LookbackMLPBlender(single_step_dim=F_dim, n_channels=C_dim, lookback_window=4, hidden=128, num_layers=2),
        "gru": GRUBlender(in_dim=F_dim, n_channels=C_dim, hidden=128, num_layers=2),
        "causal_conv": CausalConvBlender(in_dim=F_dim, n_channels=C_dim, channels=128, kernel_size=3, num_layers=3),
    }

    for name, m in blenders_init.items():
        m_path = blenders_dir / f"blender_{name}.pt"
        if m_path.exists():
            m.load_state_dict(torch.load(m_path, map_location=target_device))
            m.to(target_device)
            m.eval()
            print(f"Loaded sequence blender: {name} (V=8000)")
            models[name] = m
        else:
            print(f"Warning: {m_path.name} not found! Dynamic routing variant {name} unavailable.")

    # Interactive session parameters
    active_router = "classifier"
    decoding_mode = "soft"  # "soft", "hard", or "0", "1", "2", "3", "4"
    temp_routing = 1.0
    temp_gen = 0.5
    max_tokens = 30
    show_routing_details = True

    templates = [
        "the cat who",
        "Artificial intelligence holds",
        "def compute_features( ",
        "Formula 1 is a sport where",
    ]

    print("\n--- SAMPLE PRE-DEFINED TEMPLATES ---")
    for idx, t in enumerate(templates):
        print(f"  [{idx + 1}] {t}")
    print("------------------------------------")

    print("\nControls and Settings:")
    print("  - Type a template number (1-4) to run.")
    print("  - Type any custom prompt string to generate from it.")
    print("  - 'router <name>' to swap the active routing/blending engine. Options:")
    print("      classifier, window_mlp, lookback_mlp, gru, causal_conv")
    print("  - 'mode soft' / 'mode hard' to switch blending logic.")
    print("  - 'mode <0-4>' to force generation ONLY using a single expert:")
    print("      0: PPMI (Semantic), 1: Bigram, 2: Shape, 3: Freq, 4: Recency")
    print("  - 'temp_gen <float>' to change generation temperature (default: 0.5, 0.0 = greedy).")
    print("  - 'temp_rout <float>' to scale routing confidence temperature (default: 1.0).")
    print("  - 'max_tokens <int>' to change output length limit (default: 30).")
    print("  - 'details on/off' to toggle step-by-step routing bars (default: on).")
    print("  - 'exit' or 'quit' to exit.")

    while True:
        try:
            prompt_in = input("\nCMI-v8000 [Router: " + active_router.upper() + "]> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not prompt_in:
            continue

        prompt_lower = prompt_in.lower()

        # Handle Exit
        if prompt_lower in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Handle Router Swapping
        if prompt_lower.startswith("router "):
            val = prompt_in[7:].strip().lower()
            if val in models:
                active_router = val
                print(f"[Set] Active Sequence Router set to: {active_router.upper()}")
            else:
                print(f"[Error] Invalid router. Choose from: {list(models.keys())}")
            continue

        # Handle Settings Changes
        if prompt_lower.startswith("mode "):
            val = prompt_in[5:].strip().lower()
            if val in ["soft", "hard"] or (val.isdigit() and int(val) in range(5)):
                decoding_mode = val
                print(f"[Set] Decoding Mode set to: {decoding_mode.upper()}")
            else:
                print("[Error] Invalid mode. Choose: soft, hard, or 0-4.")
            continue

        if prompt_lower.startswith("temp_gen "):
            try:
                temp_gen = float(prompt_in[9:].strip())
                print(f"[Set] Generation Temperature set to: {temp_gen}")
            except ValueError:
                print("[Error] Must be a float.")
            continue

        if prompt_lower.startswith("temp_rout "):
            try:
                temp_routing = float(prompt_in[10:].strip())
                print(f"[Set] Routing Temperature set to: {temp_routing}")
            except ValueError:
                print("[Error] Must be a float.")
            continue

        if prompt_lower.startswith("max_tokens "):
            try:
                max_tokens = int(prompt_in[11:].strip())
                print(f"[Set] Max tokens set to: {max_tokens}")
            except ValueError:
                print("[Error] Must be an integer.")
            continue

        if prompt_lower.startswith("details "):
            val = prompt_in[8:].strip().lower()
            if val in ["on", "true", "yes"]:
                show_routing_details = True
                print("[Set] Show routing details: ON")
            elif val in ["off", "false", "no"]:
                show_routing_details = False
                print("[Set] Show routing details: OFF")
            else:
                print("[Error] Choose on or off.")
            continue

        # If numerical in 1-4, load template
        if prompt_in in ["1", "2", "3", "4"]:
            prompt_str = templates[int(prompt_in) - 1]
            print(f"Selected Template: '{prompt_str}'")
        else:
            prompt_str = prompt_in

        # Process Prompt Tokenization
        # Use BPE to encode
        prompt_encoding = bpe.encode(prompt_str)
        prompt_ids = prompt_encoding.ids

        if len(prompt_ids) == 0:
            print("[Warning] Blank prompt after BPE tokenization. Defaulting to standard token.")
            prompt_ids = [tok2id.get("the", 0)]

        print(f"\nPrompt token IDs (BPE): {prompt_ids}")
        print(f"Generating continuation (router={active_router.upper()} mode={decoding_mode.upper()} temp_gen={temp_gen} max_tokens={max_tokens}):")
        
        # Colorize and reconstruct prefix prompt
        sys.stdout.write("\x1b[36m" + prompt_str + "\x1b[0m")
        sys.stdout.flush()

        ids_seq = torch.tensor(prompt_ids, device=target_device).long()

        generated_tokens = []
        last_str_len = 0
        for step in range(max_tokens):
            next_id, routing_probs, blended_probs = generate_step(
                ids_seq, channels_lp, active_router, models, emb, V,
                decoding_mode=decoding_mode, temp_routing=temp_routing, temp_gen=temp_gen
            )

            # Append to list and sequence tensor
            generated_tokens.append(next_id)
            ids_seq = torch.cat([ids_seq, torch.tensor([next_id], device=target_device)])

            # Decode full list of newly generated tokens to avoid Unicode fragment issues
            full_str = bpe.decode(generated_tokens)
            new_part = full_str[last_str_len:]
            
            # Print new part immediately
            sys.stdout.write(new_part)
            sys.stdout.flush()
            
            last_str_len = len(full_str)

            # If details are on, we print routing information after each token
            if show_routing_details:
                sys.stdout.write("\n  ")
                for ch in range(len(ch_names)):
                    pct = routing_probs[ch].item()
                    bar = format_weights_bar(pct, width=10)
                    sys.stdout.write(f"\x1b[90m{ch_names[ch]}: {bar} {pct*100:4.1f}%\x1b[0m  ")
                sys.stdout.write("\n")
                sys.stdout.flush()

            # Stop if we hit any special end of sequence character or if token is empty
            if len(new_part) == 0 and step > 5:
                break

        print("\n\n" + "=" * 50)
        print("Generation complete!")


if __name__ == "__main__":
    main()
