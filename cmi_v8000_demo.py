"""CMI V=8000 Channels — real compiled mechanisms at wikitext scale.

Each channel produces log-prob distributions using compiled corpus statistics.
Demonstrates the CMI architecture at the real vocabulary size.
"""
import torch, torch.nn.functional as F, math, time, numpy as np
from pathlib import Path
import sys

REPO = Path("/home/drawson/llm_decoupling")
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup, load_or_build_tokens, DEVICE


def build_ppmi_channel(emb, V):
    """InstructChannel: PPMI cosine similarity — predict semantically similar tokens."""
    emb_norm = F.normalize(emb.float(), dim=1)
    sims = emb_norm @ emb_norm.T * 100.0  # (V, V)
    log_probs_all = F.log_softmax(sims, dim=1)
    return log_probs_all  # (V, V) — log P(next | current)


def build_bigram_channel(ids_np, V):
    """ReasonerChannel: bigram transition table from corpus statistics."""
    N = len(ids_np)
    bigram_counts = np.zeros((V, V), dtype=np.float64)
    count_n = min(N - 1, 10_000_000)
    for t in range(count_n):
        bigram_counts[ids_np[t], ids_np[t + 1]] += 1
    alpha = 0.01
    bigram_probs = (bigram_counts + alpha) / (bigram_counts.sum(axis=1, keepdims=True) + alpha * V)
    return torch.from_numpy(np.log(bigram_probs.clip(1e-30))).float()  # (V, V)


def build_shape_channel(bpe, V):
    """CoderChannel: token shape (syntax) transition probabilities."""
    shape_map = np.zeros(V, dtype=np.int32)
    for tid in range(V):
        s = bpe.decode([tid]).strip().lower()
        if not s: shape_map[tid] = 6
        elif s.isdigit(): shape_map[tid] = 4
        elif not any(c.isalpha() for c in s): shape_map[tid] = 5
        elif s.islower(): shape_map[tid] = 0
        elif s.isupper(): shape_map[tid] = 1
        elif s[0].isupper(): shape_map[tid] = 2
        else: shape_map[tid] = 3

    # Uniform within each shape class
    log_probs = torch.full((V, V), -float("inf"))
    for tid in range(V):
        s = shape_map[tid]
        same_shape = (shape_map == s)
        n_same = same_shape.sum()
        if n_same > 0:
            log_probs[tid, same_shape] = -math.log(n_same)
    return log_probs.float()


def build_frequency_channel(ids_np, V):
    """ToolChannel: unigram frequency — predict common tokens."""
    N = min(len(ids_np), 10_000_000)
    counts = np.bincount(ids_np[:N], minlength=V).astype(np.float64)
    alpha = 0.01
    probs = (counts + alpha) / (counts.sum() + alpha * V)
    log_probs = torch.from_numpy(np.log(probs.clip(1e-30))).float()
    return log_probs.unsqueeze(0).expand(V, V)  # same for all tokens


def build_recency_channel(ids_np, V, window=500):
    """RetrievalChannel: recency-weighted bigram — recent tokens get more weight."""
    N = min(len(ids_np), 10_000_000)
    bigram_counts = np.zeros((V, V), dtype=np.float64)
    # Use last portion of corpus for recency
    for t in range(N - window, N - 1):
        bigram_counts[ids_np[t], ids_np[t + 1]] += 1
    alpha = 0.01
    probs = (bigram_counts + alpha) / (bigram_counts.sum(axis=1, keepdims=True) + alpha * V)
    return torch.from_numpy(np.log(probs.clip(1e-30))).float()


def main():
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    emb = emb.to(DEVICE)
    ids_all = load_or_build_tokens(bpe, bpe_to_lm, V)
    ids_np = ids_all.numpy()

    print("Building 5 CMI channels at V=8000...", flush=True)

    t0 = time.time()
    ppmi_lp = build_ppmi_channel(emb, V)     # Instruct
    print(f"  PPMI semantic: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    bigram_lp = build_bigram_channel(ids_np, V)  # Reasoner
    print(f"  Bigram: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    shape_lp = build_shape_channel(bpe, V)    # Coder
    print(f"  Shape: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    freq_lp = build_frequency_channel(ids_np, V)  # Tool
    print(f"  Frequency: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    recency_lp = build_recency_channel(ids_np, V)  # Retrieval
    print(f"  Recency: {time.time()-t0:.1f}s", flush=True)

    channels_lp = [ppmi_lp, bigram_lp, shape_lp, freq_lp, recency_lp]
    ch_names = ['PPMI', 'Bigram', 'Shape', 'Freq', 'Recency']

    # Evaluate each channel on heldout
    eval_ids = ids_np[-10000:]
    targets = eval_ids[1:]
    print(f"\n{'Channel':>12} {'PPL':>10} {'Top-1':>8}")
    print("-" * 32)
    for ci, lp in enumerate(channels_lp):
        nll = 0.0; correct = 0
        for t in range(len(targets)):
            tid = eval_ids[t]
            nll += -lp[tid, targets[t]].item()
            if lp[tid].argmax().item() == targets[t]:
                correct += 1
        ppl = math.exp(nll / len(targets))
        pct = correct / len(targets) * 100
        print(f"{ch_names[ci]:>12} {ppl:>10.1f} {pct:>7.2f}%")

    print(f"\n{'Uniform':>12} {8000:>10.1f} {1/8000*100:>7.2f}%")

    # Demo: predict next token for a few prompts
    print(f"\n=== Demo predictions ===")
    demos = [
        (["the", "cat"], 0),   # Instruct: semantic prediction
        (["E", "="], 2),        # Coder: shape syntax
        (["1", "+"], 3),        # Tool: frequency
        (["the", "United"], 1), # Reasoner: bigram
    ]
    for prompt_strs, expected_ch in demos:
        prompt_ids = [tok2id.get(w, 0) for w in prompt_strs]
        print(f"\n  Prompt: {' '.join(prompt_strs)}")
        for ci, lp in enumerate(channels_lp):
            tid = prompt_ids[-1]
            top3v, top3i = lp[tid].topk(3)
            top3_strs = [bpe.decode([i.item()]) for i in top3i]
            mark = " ← expected" if ci == expected_ch else ""
            print(f"    {ch_names[ci]:>10}: {top3_strs}{mark}")


if __name__ == "__main__":
    main()
