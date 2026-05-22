"""Test harness for RetrievalChannel integration with CMI v2 capabilities.

Questions about stored documents trigger retrieval, which biases generation
toward the correct answer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from hybrid.v2_capabilities.channels import InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
from hybrid.v2_capabilities.retrieval_channel import RetrievalChannel, DEFAULT_DOCS
from hybrid.v2_capabilities.dataset import tok2id, id2tok, V, emb_dim, get_ppmi_embeddings
from hybrid.v1_blender.blender_model import build_feature_matrix
from hybrid.v3_super_blender.model import CausalConvBlender

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate(channels, blender, emb, prompt_tokens: list[str],
             max_new_tokens: int = 15, show_trace: bool = True):
    """Auto-regressive generation with 5-channel blending + retrieval."""
    current = list(prompt_tokens)
    generated = []

    print(f"\nPrompt: {' '.join(prompt_tokens)}")
    print(f"{'Step':>5} {'Token':>12} {'Weights':>50}")
    print("-" * 78)

    for step in range(max_new_tokens):
        ids = torch.tensor([tok2id[t] for t in current], device=DEVICE)
        T = len(current)

        # Run all 5 channels
        p_outputs = []
        for c in channels:
            p_outputs.append(c.forward(ids).to(DEVICE))

        # Build features per position
        all_feats = []
        for t in range(T):
            x_obs = ids[t]
            x_lag1 = ids[t - 1] if t > 0 else torch.zeros_like(x_obs)

            log_p_obs_t = torch.stack([p[t, x_obs] for p in p_outputs])
            log_p_lag1_t = torch.stack([p[t, x_lag1] for p in p_outputs])

            entropy_t = []
            max_log_prob_t = []
            for p in p_outputs:
                p_dist = p[t].exp()
                entropy_t.append(-(p_dist * p[t]).sum())
                max_log_prob_t.append(p[t].max())

            feat = build_feature_matrix(
                log_p_obs_t.unsqueeze(0),
                log_p_lag1_t.unsqueeze(0),
                torch.stack(entropy_t).unsqueeze(0),
                torch.stack(max_log_prob_t).unsqueeze(0),
                emb.to(DEVICE),
                x_obs.unsqueeze(0),
                use_embedding=True,
            )
            all_feats.append(feat)

        features = torch.cat(all_feats, dim=0).to(DEVICE)

        with torch.no_grad():
            log_w = blender(features.unsqueeze(0)).squeeze(0)

        latest_w = log_w[-1].exp()

        # Blend
        blended = torch.zeros(V, device=DEVICE)
        for c_idx in range(len(channels)):
            blended += latest_w[c_idx] * p_outputs[c_idx][-1].exp()

        next_id = blended.argmax().item()
        next_tok = id2tok[next_id]
        generated.append(next_tok)
        current.append(next_tok)

        if show_trace:
            w_str = " ".join(f"{latest_w[i]:.2f}" for i in range(len(channels)))
            print(f"{step:5d} {next_tok:>12} {w_str}")

    result = " ".join(generated)
    print(f"\nResult: {result}")
    return result


def main():
    emb = get_ppmi_embeddings().to(DEVICE)

    # Initialize capability channels
    instruct = InstructChannel(tok2id, id2tok, emb)
    reasoner = ReasonerChannel(tok2id, id2tok)
    coder = CoderChannel(tok2id, id2tok)
    tool = ToolChannel(tok2id, id2tok)
    retrieval = RetrievalChannel(tok2id, id2tok, emb, doc_texts=DEFAULT_DOCS)

    channels = [instruct, reasoner, coder, tool, retrieval]
    C = len(channels)
    F = 4 * C + emb_dim  # 4*5 + 16 = 36

    # Load pre-trained blender or use random init
    blender = CausalConvBlender(in_dim=F, n_channels=C, channels=64, kernel_size=3).to(DEVICE)
    bpath = REPO / "hybrid/v2_capabilities/super_blender_causal_conv_5ch.pt"
    if bpath.exists():
        ckpt = torch.load(bpath, map_location=DEVICE, weights_only=False)
        if ckpt.get('n_channels') == C:
            blender.load_state_dict(ckpt['state_dict'])
            print(f"Loaded {ckpt.get('n_channels')}-channel blender")
        else:
            print(f"Blender trained for {ckpt.get('n_channels')}, we have {C} — random init")
    else:
        print("No trained blender — retrieval still works, routing is uniform")

    blender.eval()

    print("=" * 60)
    print("RETRIEVAL CHANNEL TEST — 5-Channel CMI Generation")
    print("=" * 60)

    # Test 1: Retrieval question — answer is in document
    print("\n[TEST 1] Retrieval from stored documents")
    generate(channels, blender, emb,
             ["What", "is", "a", "cat"])
    generate(channels, blender, emb,
             ["explain", "gravity"])
    generate(channels, blender, emb,
             ["translate", "dog", "to", "french"])

    # Test 2: Translation (InstructChannel)
    print("\n[TEST 2] Translation — should route to InstructChannel")
    generate(channels, blender, emb,
             ["translate", "dog", "to", "french"])

    # Test 3: Reasoning (ReasonerChannel)
    print("\n[TEST 3] Transitive reasoning — should route to ReasonerChannel")
    generate(channels, blender, emb,
             ["E0001", "is", "larger", "than", "E0002", ".",
              "E0002", "is", "larger", "than", "E0003", ".",
              "Therefore", ",", "E0001", "is", "larger", "than"])

    # Test 4: Code generation (CoderChannel)
    print("\n[TEST 4] Code completion — should route to CoderChannel")
    generate(channels, blender, emb,
             ["def", "get_sum", "(", "a", ",", "b", ")", ":",
              "return", "a", "+"])

    # Test 5: Tool use (ToolChannel)
    print("\n[TEST 5] Calculator — should route to ToolChannel")
    generate(channels, blender, emb,
             ["What", "is", "12", "+", "15", "?", "[USE_TOOL:",
              "calculator", "expr=", "12+15", "]", "Answer", "is"])

    # Test 6: Retrieval + cross-channel — does retrieval boost Instruct?
    print("\n[TEST 6] Retrieval of French facts → should help translation")
    generate(channels, blender, emb,
             ["translate", "apple", "to", "french"])
    # Test 7: Retrieval of gravity facts
    print("\n[TEST 7] Retrieval of physics facts")
    generate(channels, blender, emb,
             ["explain", "gravity"])


if __name__ == "__main__":
    main()
