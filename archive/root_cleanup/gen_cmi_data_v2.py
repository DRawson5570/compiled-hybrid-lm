"""CMI Training Data Generator — V=8000 with realistic synthetic channel outputs.

Each channel produces a domain-biased log-prob distribution.
The blender learns to route based on per-channel statistics.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, random, math, pickle
from pathlib import Path
import sys

REPO = Path("/home/drawson/llm_decoupling")
sys.path.insert(0, str(REPO))
from compile_wiki_lm_v13 import load_setup, load_or_build_tokens
DS = Path("/home/drawson/deepseek_experiments")


def classify_token(tid: int, bpe, V: int) -> list[int]:
    """Returns which capability channels this token is relevant for (0-4 bits)."""
    s = bpe.decode([tid]).strip().lower()

    code_kw = {'def', 'return', 'import', 'class', 'if', 'else', 'for', 'while',
               'try', 'except', 'with', 'as', 'from', 'lambda', 'pass', 'print',
               'range', 'len', 'int', 'str', 'list', 'dict', 'set', 'none',
               'true', 'false', 'and', 'or', 'not', 'in', 'is', '+', '-', '*', '/'}
    instruct_kw = {'translate', 'explain', 'what', 'how', 'why', 'when', 'who',
                   'describe', 'define', 'convert', 'calculate', 'find', 'list',
                   'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have',
                   'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from',
                   'this', 'that', 'it', 'be', 'can', 'will', 'would', 'should'}
    number_strs = set('0123456789')

    channels = [0, 0, 0, 0, 0]  # bit per channel

    if s in code_kw:
        channels[2] = 1  # Coder
        channels[0] = 1  # also Instruct (overlap)
    if s in instruct_kw:
        channels[0] = 1  # Instruct
    if s and s[0].isupper() and len(s) > 1:
        channels[1] = 1  # Reasoner (entities/proper nouns)
        channels[4] = 1  # Retrieval
    if any(c in number_strs for c in s) and s not in instruct_kw:
        channels[3] = 1  # Tool
    if not any(channels):
        channels[0] = 1  # default: Instruct-like common text

    return channels


def make_channel_log_probs(ids_t, bpe, V, d, emb, ch_idx):
    """Generate realistic synthetic log-probs for a specific channel."""
    T = len(ids_t)
    device = ids_t.device

    # Classify each token for this channel
    relevance = torch.tensor([classify_token(int(ids_t[t]), bpe, V)[ch_idx]
                              for t in range(T)], device=device, dtype=torch.float32)

    # Base: uniform distribution
    log_probs = torch.full((T, V), -math.log(V), device=device)

    # Boost relevant tokens
    for t in range(T):
        if relevance[t] > 0:
            # Boost tokens that are relevant to this channel
            for tid in range(V):
                if classify_token(tid, bpe, V)[ch_idx]:
                    log_probs[t, tid] = 0.0  # high prob
            # Re-normalize
            log_probs[t] = log_probs[t] - torch.logsumexp(log_probs[t], dim=0)

    return log_probs


def classify_window(token_ids, bpe, window_size=20):
    """Classify a token ID window into capability type based on content."""
    strs = [bpe.decode([int(tid)]) for tid in token_ids[:window_size]]
    joined = " ".join(strs).lower()

    code_keywords = {'def', 'return', 'import', 'class', 'if', 'else', 'for', 'while',
                     'try', 'except', 'with', 'as', 'from', 'lambda', 'pass', 'print',
                     'range', 'len', 'int', 'str', 'list', 'dict', 'set', 'none'}
    code_score = sum(1 for s in strs if s.strip().lower() in code_keywords)

    has_numbers = any(s.strip().isdigit() for s in strs)
    has_operators = any(s.strip() in {'+', '-', '*', '/', '=', '<', '>', '?'} for s in strs)
    has_entities = sum(1 for s in strs if s.strip() and s[0].isupper() and len(s) > 2)

    instruct_words = {'translate', 'explain', 'what', 'how', 'why', 'when', 'who',
                      'describe', 'define', 'convert', 'calculate', 'find', 'list'}
    has_instruct = any(s.strip().lower() in instruct_words for s in strs)

    if code_score >= 2:
        return 2
    elif has_numbers and has_operators:
        return 3
    elif has_entities >= 3:
        return 1
    elif has_instruct:
        return 0
    elif has_entities >= 1:
        return 4
    else:
        return 0


def main():
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    emb = emb.float()
    ids_all = load_or_build_tokens(bpe, bpe_to_lm, V)
    ids_np = ids_all.numpy()
    total_tokens = len(ids_np)

    n_windows = 10000
    window_size = 32
    stride = max(1, (total_tokens - window_size) // n_windows)

    X, y = [], []
    names = ['instruct', 'reasoner', 'coder', 'tool', 'retrieval']
    C = 5
    F = 4 * C + d  # 20 + 256 = 276

    # Precompute: which tokens are relevant for each channel
    print("Precomputing token-channel mapping...", flush=True)
    t0 = time.time()
    tok_channels = np.zeros((V, C), dtype=np.int8)
    for tid in range(V):
        chs = classify_token(tid, bpe, V)
        for ci in range(C):
            tok_channels[tid, ci] = chs[ci]
    # Precompute channel-specific token sets as boolean masks
    ch_masks = [torch.tensor(tok_channels[:, ci].astype(bool), dtype=torch.bool)
                for ci in range(C)]
    print(f"  done: {time.time()-t0:.1f}s", flush=True)

    print(f"Generating {n_windows} windows (V={V}, F={F})...", flush=True)
    t0 = time.time()

    for i in range(n_windows):
        start = i * stride
        if start + window_size >= total_tokens:
            break
        window_ids = ids_np[start:start + window_size]
        window_label = classify_window(window_ids, bpe, window_size)

        ids_t = torch.from_numpy(window_ids).long()
        t = window_size - 1
        x_o = ids_t[t]
        x_l1 = ids_t[t - 1] if t > 0 else torch.zeros_like(ids_t[t])

        feat_parts = []
        for ch in range(C):
            # Channel output: uniform base, boost relevant tokens
            lp = torch.full((V,), -math.log(V), dtype=torch.float32)
            lp[ch_masks[ch]] = 0.0
            lp = lp - torch.logsumexp(lp, dim=0)
            # Only need last position's stats
            feat_parts.append(lp[x_o].unsqueeze(0))  # log_p_observed
        for ch in range(C):
            lp = torch.full((V,), -math.log(V), dtype=torch.float32)
            lp[ch_masks[ch]] = 0.0
            lp = lp - torch.logsumexp(lp, dim=0)
            feat_parts.append(lp[x_l1].unsqueeze(0))  # log_p_lag1
        for ch in range(C):
            lp = torch.full((V,), -math.log(V), dtype=torch.float32)
            lp[ch_masks[ch]] = 0.0
            lp = lp - torch.logsumexp(lp, dim=0)
            p = lp.exp()
            feat_parts.append(-(p * lp).sum().unsqueeze(0))  # entropy
        for ch in range(C):
            lp = torch.full((V,), -math.log(V), dtype=torch.float32)
            lp[ch_masks[ch]] = 0.0
            lp = lp - torch.logsumexp(lp, dim=0)
            feat_parts.append(lp.max().unsqueeze(0))  # max_log_prob
        feat_parts.append(emb[x_o])

        feat = torch.cat(feat_parts)
        X.append(feat.numpy())
        y.append(window_label)

        if i % 2000 == 0:
            print(f"  {i}/{n_windows} ({time.time()-t0:.0f}s)", flush=True)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Done: {len(X)} examples, F={F}", flush=True)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Done: {len(X)} examples, F={F}", flush=True)

    for ci, name in enumerate(names):
        print(f"  {name}: {(y == ci).sum()}")

    out_path = DS / "artifacts" / "cmi_train_8k.npz"
    out_path.parent.mkdir(exist_ok=True)
    np.savez(out_path, X=X, y=y)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
