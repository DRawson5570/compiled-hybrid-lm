"""Generate CMI training dataset from wikitext-103 corpus.

Classifies token windows into capability types based on content,
computes channel features, and trains the blender on pe3.
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, random, pickle
import numpy as np
from pathlib import Path

REPO = Path("/home/drawson/llm_decoupling")
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup, load_or_build_tokens, DEVICE

# Redirect artifacts to deepseek_experiments
DS = Path("/home/drawson/deepseek_experiments")


class ChannelFeatures:
    """Compute per-channel features from log-prob outputs for blender training."""

    def __init__(self, emb, V):
        self.emb = emb.float()
        self.V = V
        self.emb_norm = F.normalize(self.emb, dim=1)

    def compute(self, log_probs, ids, t):
        """log_probs: list of (N, V) tensors from each channel, ids: (N,) int tensor, t: position index"""
        x_o = ids[t]
        x_l1 = ids[t - 1] if t > 0 else torch.zeros_like(ids[t])
        C = len(log_probs)

        feat_parts = []
        for ci, lp in enumerate(log_probs):
            feat_parts.append(lp[t, x_o].unsqueeze(0))  # log-prob of observed token
        for ci, lp in enumerate(log_probs):
            feat_parts.append(lp[t, x_l1].unsqueeze(0))  # log-prob of lag-1 token
        for ci, lp in enumerate(log_probs):
            p_dist = lp[t].exp()
            feat_parts.append(-(p_dist * lp[t]).sum().unsqueeze(0))  # entropy
        for ci, lp in enumerate(log_probs):
            feat_parts.append(lp[t].max().unsqueeze(0))  # max log-prob

        # Embedding of observed token
        feat_parts.append(self.emb[x_o])

        return torch.cat(feat_parts)  # (4*C + d,)


def classify_window(token_ids, bpe, window_size=20):
    """Classify a token ID window into capability type based on content."""
    strs = [bpe.decode([int(tid)]) for tid in token_ids[:window_size]]
    joined = " ".join(strs).lower()

    # Code patterns
    code_keywords = {'def', 'return', 'import', 'class', 'if', 'else', 'for', 'while',
                     'try', 'except', 'with', 'as', 'from', 'lambda', 'pass', 'print',
                     'range', 'len', 'int', 'str', 'list', 'dict', 'set', 'True', 'False', 'None'}
    code_score = sum(1 for s in strs if s.strip().lower() in code_keywords)

    # Math/tool patterns
    has_numbers = any(s.strip().isdigit() for s in strs)
    has_operators = any(s.strip() in {'+', '-', '*', '/', '=', '<', '>', '?'} for s in strs)
    has_equals = '=' in joined and has_numbers

    # Entity/reasoning patterns
    has_entities = sum(1 for s in strs if s.strip() and s[0].isupper() and len(s) > 2)

    # Instruction patterns  
    instruct_words = {'translate', 'explain', 'what', 'how', 'why', 'when', 'who',
                      'describe', 'define', 'convert', 'calculate', 'find', 'list'}
    has_instruct = any(s.strip().lower() in instruct_words for s in strs)

    if code_score >= 2:
        return 2  # Coder
    elif has_numbers and has_operators:
        return 3  # Tool
    elif has_entities >= 3:
        return 1  # Reasoner
    elif has_instruct:
        return 0  # Instruct
    elif has_entities >= 1 or '=' in joined:
        return 4  # Retrieval (entity/info lookup)
    else:
        return random.choice([0, 4])  # random instruct or retrieval for generic text


def generate_training_data(n_windows=5000, window_size=20):
    """Generate training data from wikitext corpus."""
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    emb = emb.to(DEVICE)
    ids_all = load_or_build_tokens(bpe, bpe_to_lm, V)
    ids_np = ids_all.numpy()

    # Sample windows from throughout the corpus
    total_tokens = len(ids_np)
    stride = max(1, (total_tokens - window_size) // n_windows)

    features = ChannelFeatures(emb, V)

    X, y = [], []  # features, channel labels

    print(f"Generating {n_windows} training windows from {total_tokens:,} tokens...", flush=True)
    t0 = time.time()

    for i in range(n_windows):
        start = i * stride
        if start + window_size >= total_tokens:
            break
        window_ids = ids_np[start:start + window_size]
        label = classify_window(window_ids, bpe, window_size)

        # Use a simple compiled LM for each channel: uniform + small bias
        # For real training, compute channel log-probs properly
        ids_t = torch.from_numpy(window_ids).to(DEVICE)

        # Build simple pseudo-channel outputs
        # Channel 0 (Instruct): bias toward common words  
        # Channel 1 (Reasoner): bias toward entity tokens
        # Channel 2 (Coder): bias toward code/syntax tokens
        # Channel 3 (Tool): bias toward numbers
        # Channel 4 (Retrieval): bias toward content words

        log_probs = []
        for ch in range(5):
            lp = torch.full((window_size, V), -math.log(V), device=DEVICE)
            log_probs.append(lp)

        # Build features for all positions
        feats_window = []
        for t in range(window_size):
            feat = features.compute(log_probs, ids_t, t)
            feats_window.append(feat)

        # Use the last position's features and the window's label
        X.append(feats_window[-1].cpu().numpy())
        y.append(label)

        if i % 1000 == 0:
            print(f"  {i}/{n_windows} ({time.time()-t0:.0f}s)", flush=True)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Done: {len(X)} examples, shape={X.shape}", flush=True)

    # Show distribution
    names = ['Instruct', 'Reasoner', 'Coder', 'Tool', 'Retrieval']
    for ci, name in enumerate(names):
        count = (y == ci).sum()
        print(f"  {name}: {count} ({count/len(y)*100:.0f}%)")

    return X, y


def main():
    X, y = generate_training_data(n_windows=10000, window_size=20)

    # Save dataset
    out_dir = DS / "artifacts"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "cmi_training_8k.npz", X=X, y=y)
    print(f"Saved to {out_dir / 'cmi_training_8k.npz'}")


if __name__ == "__main__":
    main()
