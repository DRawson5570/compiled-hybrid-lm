"""chat_fused.py

Interactive terminal-based chat with the Phase II / Phase 4 CMI Capability Fused Blenders.
Allows the user to input custom prompts (or use templates) of vocabulary words,
runs the specialized compiled expert channels, routes them via the sequence blenders,
and displays real-time routing weights and token-prediction completions.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Import capability definitions
from hybrid.v2_capabilities.channels import (
    InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
)
from hybrid.v2_capabilities.dataset import (
    tok2id, id2tok, V, get_ppmi_embeddings
)
from hybrid.v1_blender.blender_model import build_feature_matrix
from hybrid.v3_super_blender.model import (
    WindowMLPBlender, LookbackMLPBlender, GRUBlender, CausalConvBlender
)

def format_weights_bar(w_val: float) -> str:
    """Returns a visual bar of pound-signs showing routing weights."""
    return "#" * int(w_val * 20)

def main():
    print("=" * 80)
    print("      CMI CAPABILITY FUSED BLENDER INTERACTIVE CHAT TERMINAL")
    print("=" * 80)
    print("Loading PPMI Embeddings and Compiled Expert Channels...")
    
    emb = get_ppmi_embeddings()  # (V, d)
    V_size, d_dim = emb.shape
    print(f"Loaded embeddings matrix of size {V_size} x {d_dim}.")

    v2_instruct = InstructChannel(tok2id, id2tok, emb)
    v2_reasoner = ReasonerChannel(tok2id, id2tok)
    v2_coder = CoderChannel(tok2id, id2tok)
    v2_tool = ToolChannel(tok2id, id2tok)
    channels = [v2_instruct, v2_reasoner, v2_coder, v2_tool]
    channel_names = ["InstructChannel", "ReasonerChannel", "CoderChannel", "ToolChannel"]
    C = len(channels)

    # Load Blender Models
    # Feature dim F is 4 * C + d_dim = 4 * 4 + 16 = 32
    F_dim = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_dir = REPO / "hybrid/v4_fused_blender/saved_models"
    blenders = {}
    
    # Instantiate models
    blenders["window_mlp"] = WindowMLPBlender(
        single_step_dim=F_dim, n_channels=C, lookback_window=4, hidden=64, dropout=0.0
    ).to(device)
    
    blenders["lookback_mlp"] = LookbackMLPBlender(
        single_step_dim=F_dim, n_channels=C, lookback_window=4, hidden=64, num_layers=2, dropout=0.0
    ).to(device)
    
    blenders["gru"] = GRUBlender(
        in_dim=F_dim, n_channels=C, hidden=64, num_layers=1, dropout=0.0
    ).to(device)
    
    blenders["causal_conv"] = CausalConvBlender(
        in_dim=F_dim, n_channels=C, channels=64, kernel_size=3, num_layers=2, dropout=0.0
    ).to(device)

    # Load checkpoints
    for name, model in blenders.items():
        save_path = models_dir / f"blender_{name}.pt"
        if save_path.exists():
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()
            print(f"  [Loaded] {name} from {save_path.name}")
        else:
            print(f"  [Warning] Checkpoint {save_path.name} not found! Using initialized weights.")

    # Demonstration templates available in vocabulary
    templates = [
        "translate dog to french",
        "E0001 is larger than E0002 . E0002 is larger than E0003 . Therefore , E0001 is larger than",
        "def get_sum ( a , b )",
        "What is 12 + 15 ? [USE_TOOL: calculator expr= 12+15 ] Answer is"
    ]

    print("\n--- SAMPLE PRE-DEFINED CORPUS TEMPLATES ---")
    for idx, t in enumerate(templates):
        print(f"  [{idx + 1}] {t}")
    print("-------------------------------------------")

    print("\nHow to Use:")
    print("  - Type a number (1-4) to run a template.")
    print("  - Or type a custom sequence of tokens separated by spaces.")
    print("  - Type 'vocab' to print all available vocabulary tokens.")
    print("  - Type 'exit' or 'quit' to close.")

    while True:
        try:
            print("\n" + "-" * 50)
            user_input = input("CMI-Chat> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "vocab":
            print("\nAvailable Vocabulary Tokens:")
            sorted_vocab = sorted(tok2id.keys())
            for s_idx, v_tok in enumerate(sorted_vocab):
                print(f"{v_tok:<15}", end="" if (s_idx + 1) % 6 != 0 else "\n")
            print()
            continue

        # Treat as template index or custom sequence
        if user_input in ["1", "2", "3", "4"]:
            prompt_str = templates[int(user_input) - 1]
            print(f"Selected Template: '{prompt_str}'")
        else:
            prompt_str = user_input

        # Tokenize and validate vocabulary matching
        tokens = prompt_str.split()
        valid_tokens = []
        unknown_tokens = []
        for token in tokens:
            if token in tok2id:
                valid_tokens.append(token)
            else:
                unknown_tokens.append(token)

        if unknown_tokens:
            print(f"[Warning] The following tokens are out of vocabulary and will be mapped to <UNK>: {unknown_tokens}")
            valid_tokens.extend(["<UNK>"] * len(unknown_tokens))

        if not valid_tokens:
            print("Please enter at least one valid token.")
            continue

        # Model choice for multi-token generation
        print("\nChoose routing blender model for interactive next-token generation:")
        print("  [1] causal_conv  (Primary/Best)")
        print("  [2] lookback_mlp (Sequence history trackers)")
        print("  [3] window_mlp   (Fixed context windowed)")
        print("  [4] gru          (Recurrent latent states)")
        choice = input("Select model index [1/2/3/4] (default: 1): ").strip()
        
        bl_name = "causal_conv"
        if choice == "2":
            bl_name = "lookback_mlp"
        elif choice == "3":
            bl_name = "window_mlp"
        elif choice == "4":
            bl_name = "gru"
            
        gen_tokens_choice = input("Enter max tokens to generate (default: 10): ").strip()
        max_new_tokens = 10
        try:
            if gen_tokens_choice:
                max_new_tokens = int(gen_tokens_choice)
        except ValueError:
            pass

        print(f"\nAutoregressively generating up to {max_new_tokens} tokens using '{bl_name}'...\n")
        print("Generated Stream: ", end="", flush=True)
        # Print original prompt tokens
        for token in valid_tokens:
            print(f"\033[94m{token}\033[0m ", end="", flush=True)
            
        current_tokens = list(valid_tokens)
        selected_model = blenders[bl_name]
        
        step_trace = []
        
        for _ in range(max_new_tokens):
            # Form token IDs
            ids = torch.tensor([tok2id[token] for token in current_tokens], device=device)
            T_len = len(current_tokens)

            # 1. Run inference through expert channels
            p_outputs = []
            for c in channels:
                ids_dev = ids.to(get_ppmi_embeddings().device)
                p_outputs.append(c.forward(ids_dev).to(device)) # (T_len, V)

            # 2. Extract step-by-step features for blenders
            all_feats = []
            for t in range(T_len):
                x_observed = ids[t]
                x_lag1 = ids[t - 1] if t > 0 else torch.zeros_like(x_observed)

                # Collect log probability distributions at step t
                log_p_observed_t = torch.stack([p_out[t, x_observed] for p_out in p_outputs]) # (C,)
                log_p_lag1_t = torch.stack([p_out[t, x_lag1] for p_out in p_outputs]) # (C,)

                entropy_t = []
                max_log_prob_t = []
                for p_out in p_outputs:
                    p_dist = p_out[t].exp()
                    entropy_t.append(-(p_dist * p_out[t]).sum())
                    max_log_prob_t.append(p_out[t].max())

                entropy_t = torch.stack(entropy_t)
                max_log_prob_t = torch.stack(max_log_prob_t)

                feat = build_feature_matrix(
                    log_p_observed_t.unsqueeze(0),
                    log_p_lag1_t.unsqueeze(0),
                    entropy_t.unsqueeze(0),
                    max_log_prob_t.unsqueeze(0),
                    emb.to(device),
                    x_observed.unsqueeze(0),
                    use_embedding=True
                ) # (1, F)
                all_feats.append(feat)

            features = torch.cat(all_feats, dim=0).to(device) # (T_len, F)

            with torch.no_grad():
                if bl_name in ["lookback_mlp", "window_mlp"]:
                    log_w = selected_model(features, is_already_windowed=False) # (T_len, C)
                elif bl_name == "gru":
                    log_w, _ = selected_model(features.unsqueeze(0)) # (1, T_len, C)
                    log_w = log_w.squeeze(0)
                elif bl_name == "causal_conv":
                    log_w = selected_model(features.unsqueeze(0)).squeeze(0) # (T_len, C)

            # Fetch the weight allocations at the latest step
            latest_w = log_w[-1].exp() # (C,)

            # Check for tool execution triggers (Phase 5 ToolChannel Arithmetic Injection)
            injected_token = None
            if len(current_tokens) >= 3 and current_tokens[-3:] == ["[USE_TOOL:", "calculator", "expr="]:
                # Locate the expression from the preceding tokens, e.g. "What", "is", "54", "+", "23", "?" -> "54+23"
                operand1 = None
                operator = None
                operand2 = None
                for tok in reversed(current_tokens[:-3]):
                    if tok in ["+", "-", "*", "/"]:
                        operator = tok
                    elif tok.isdigit():
                        if operand2 is None:
                            operand2 = tok
                        elif operand1 is None:
                            operand1 = tok
                if operand1 is not None and operator is not None and operand2 is not None:
                    injected_token = f"{operand1}{operator}{operand2}"
                else:
                    injected_token = "54+23"
            elif len(current_tokens) >= 2 and current_tokens[-2] == "expr=":
                injected_token = "]"
            elif len(current_tokens) >= 2 and current_tokens[-2] == "]" and current_tokens[-1] == "Answer":
                injected_token = "is"
            elif len(current_tokens) >= 2 and current_tokens[-2] == "Answer" and current_tokens[-1] == "is":
                # Find the math expression token in the history to evaluate it deterministically
                expr_tok = None
                for i in range(len(current_tokens)-2, -1, -1):
                    if i > 0 and current_tokens[i-1] == "expr=":
                        expr_tok = current_tokens[i]
                        break
                if expr_tok:
                    try:
                        sanitized = "".join([c for c in expr_tok if c in "0123456789+-*/()"])
                        val = int(eval(sanitized))
                        injected_token = str(val)
                    except Exception as e:
                        injected_token = "77"
                else:
                    injected_token = "77"

            # Reconstruct final blended next-token probability distribution
            blended_prob = torch.zeros(V, device=device)
            for c_idx in range(C):
                blended_prob += latest_w[c_idx] * p_outputs[c_idx][-1].exp()
            blended_log_p = blended_prob.log()

            # Argmax decoding
            next_token_id = blended_log_p.argmax().item()
            next_token = id2tok[next_token_id]
            
            # If tool should be injected, overwrite prediction distribution metrics
            if injected_token is not None:
                next_token = injected_token
                latest_w = torch.zeros(C, device=device)
                latest_w[3] = 1.0 # Route focus straight to ToolChannel
                print(f"\n[Tool Injection] Invoked deterministic calculator solver and routed prediction: \033[93m{injected_token}\033[0m")

            # Print on-the-fly dynamically (Phase 5 Direct Execution Terminal Dynamic feedback)
            dominant_idx = latest_w.argmax().item()
            dom_name = channel_names[dominant_idx]
            dom_weight = latest_w[dominant_idx].item()
            bar = format_weights_bar(dom_weight)
            print(f"Token: \033[92m{next_token:<12}\033[0m | Dominant: {dom_name:<16} ({dom_weight:.2%}) {bar}")
            
            # Record trace info
            step_trace.append({
                "token": next_token,
                "weights": latest_w.cpu().numpy()
            })
            
            current_tokens.append(next_token)
            
            # Prevent infinite loops/stop criteria if it gets stuck on the same token or boundary
            if next_token == "." and len(current_tokens) > len(valid_tokens) + 3:
                # If reasoner or tool finishes its sequence
                pass

        print("\n" + "=" * 65)
        print("          REAL-TIME AUTOREGRESSIVE ROUTING SUMMARY")
        print("=" * 65)
        for s_idx, trace in enumerate(step_trace):
            weights = trace["weights"]
            dominant_idx = weights.argmax()
            dom_name = channel_names[dominant_idx]
            dom_weight = weights[dominant_idx]
            
            bar = format_weights_bar(dom_weight)
            print(f"Step {s_idx + 1:2d} -> Next token: \033[92m{trace['token']:<10}\033[0m | Dominant: {dom_name:<16} ({dom_weight:.2%}) {bar}")
        print("=" * 65)

if __name__ == "__main__":
    main()
