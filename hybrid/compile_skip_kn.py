"""SkipGram KN Channel — gapped Kneser-Ney n-gram model.

Implements three gapped patterns with modified KN (absolute discounting, backoff to unigram):
  P1: (w_{-3}, *, w_{-1}) → t    [2-gram with 1 gap]
  P2: (w_{-4}, *, *, w_{-1}) → t  [2-gram with 2 gaps]
  P3: (w_{-4}, *, w_{-2}, w_{-1}) → t  [3-gram with 1 gap]

Interface: .score(ids) → (N, V) float32 log-probs per position.
Matches dtype/shape conventions from ModifiedKNGram in compile_wiki_lm_v23.py.
"""
from __future__ import annotations

import math, time, pickle, sys
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

REPO = Path("/home/drawson/llm_decoupling")
ARTIFACT_DIR = REPO / "artifacts/skip_kn_v1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
V = 8000


def count_skip_patterns(ids: np.ndarray, gap_specs: list[tuple[tuple[int, ...], int]]):
    """
    Count gapped patterns in the token stream.
    
    Args:
        ids: (N,) int64 token IDs
        gap_specs: list of ((context_positions,), order) tuples
          e.g. ((-3, -1), 2) for pattern P1: ids[t-3], ids[t-1] → ids[t]
    
    Returns:
        counts: dict mapping (context_tuple, order) -> Counter of target tokens
    """
    counts = {}
    N = len(ids)
    
    for spec in gap_specs:
        ctx_positions, order = spec
        max_lookback = max(abs(p) for p in ctx_positions)
        ctx_counts = defaultdict(lambda: defaultdict(int))
        
        for t in range(max_lookback, N):
            ctx = tuple(ids[t + p] for p in ctx_positions)  # e.g. ids[t-3], ids[t-1]
            target = ids[t]
            ctx_counts[ctx][target] += 1
        
        counts[(ctx_positions, order)] = dict(ctx_counts)
        print(f"  Pattern {ctx_positions}: {len(ctx_counts):,} unique contexts", flush=True)
    
    return counts


def compute_kn_params(counts_dict: dict, V: int, order: int):
    """
    Compute Kneser-Ney parameters with conservative absolute discounting.
    Uses fixed D=0.5 for all counts to avoid overly aggressive discounting
    that dilutes pattern signal into the unigram backoff.
    """
    probs = {}
    backoff_weights = {}
    
    D1 = D2 = D3p = 0.5  # conservative fixed discount
    
    for ctx, token_counts in counts_dict.items():
        total = sum(token_counts.values())
        p = {}
        lam = 0.0
        
        for token, c in token_counts.items():
            d = D1
            p[token] = max(c - d, 0) / total
            lam += d / total
        
        probs[ctx] = p
        backoff_weights[ctx] = lam
    
    print(f"    Fixed D=0.5, {len(counts_dict):,} contexts", flush=True)
    return probs, backoff_weights, (D1, D2, D3p)


class SkipGramKNChannel:
    """Gapped Kneser-Ney language model channel."""
    
    def __init__(self):
        self.V = V
        self.patterns = []       # list of (probs, backoff, ctx_positions, name)
        self.unigram = None      # (V,) float64 unigram continuation probs
    
    def build(self, ids: np.ndarray):
        """
        Build count tables and KN parameters from token stream.
        
        Args:
            ids: (N,) int64 numpy array of token IDs
        """
        N = len(ids)
        print(f"[skip_kn] Building from {N:,} tokens...", flush=True)
        t0 = time.time()
        
        # Define gapped patterns: ((context_positions), order)
        gap_specs = [
            ((-3, -1), 2),          # P1: w_{-3}, *, w_{-1} → t
            ((-4, -1), 2),          # P2: w_{-4}, *, *, w_{-1} → t
            ((-4, -2, -1), 3),      # P3: w_{-4}, *, w_{-2}, w_{-1} → t
        ]
        names = ["P1(skip1)", "P2(skip2)", "P3(skip1-3g)"]
        
        # Count patterns
        print("[count] Counting gapped patterns...", flush=True)
        raw_counts = count_skip_patterns(ids, gap_specs)
        
        # Compute unigram continuation counts (for KN backoff)
        print("[uni] Computing unigram continuation...", flush=True)
        uni_cont = np.zeros(V, dtype=np.float64)
        # From P1 (bigram-like): count unique left contexts per right token
        for i, spec in enumerate(gap_specs):
            ctx_positions, order = spec
            counts = raw_counts[(ctx_positions, order)]
            for ctx, token_counts in counts.items():
                for token in token_counts:
                    uni_cont[token] += 1.0
        
        uni_total = uni_cont.sum()
        self.unigram = (uni_cont + 1e-9) / (uni_total + V * 1e-9)
        print(f"    unigram done, total conts={uni_total:.0f}", flush=True)
        
        # Build KN probs for each pattern
        print("[kn] Computing KN probabilities...", flush=True)
        for i, spec in enumerate(gap_specs):
            ctx_positions, order = spec
            counts = raw_counts[(ctx_positions, order)]
            probs, backoff, discounts = compute_kn_params(counts, V, order)
            self.patterns.append({
                'probs': probs,
                'backoff': backoff,
                'ctx_positions': ctx_positions,
                'name': names[i],
                'discounts': discounts,
            })
        
        print(f"[build] total {time.time()-t0:.1f}s", flush=True)
    
    def score(self, ids: np.ndarray) -> np.ndarray:
        """
        Compute per-position log-probabilities using interpolation.
        Each pattern contributes its discounted probability, weighted by
        the total probability mass it captures. Unseen patterns contribute zero.
        Interpolation ensures even sparse patterns add signal.
        """
        N = len(ids)
        log_probs = np.full((N, V), -math.log(V), dtype=np.float32)
        
        min_lookback = max(abs(p) for pat in self.patterns for p in pat['ctx_positions'])
        
        for t in range(min_lookback, N):
            # Interpolated probability vector
            p_interp = np.zeros(V, dtype=np.float64)
            total_weight = 0.0
            
            # Try each pattern, add its contribution weighted by how much mass it captures
            for pat in self.patterns:
                ctx_positions = pat['ctx_positions']
                ctx = tuple(ids[t + p] for p in ctx_positions)
                
                if ctx in pat['probs']:
                    prob_dict = pat['probs'][ctx]
                    backoff_wt = pat['backoff'][ctx]
                    # This pattern explains (1 - backoff_wt) of the probability mass
                    mass = 1.0 - backoff_wt
                    for tok, prob in prob_dict.items():
                        p_interp[tok] += prob  # already mass-weighted in prob_dict
                    p_interp += backoff_wt * self.unigram
                    total_weight = 1.0
                    break  # use first matching pattern (highest order)
            
            if total_weight == 0.0:
                # No pattern matched — use unigram
                p_interp = self.unigram.copy()
            
            # Re-normalize
            s = p_interp.sum()
            if s > 0:
                p_interp /= s
            else:
                p_interp = np.ones(V) / V
            
            log_probs[t] = np.log(p_interp.clip(1e-30)).astype(np.float32)
        
        # Positions before min_lookback: use unigram
        log_probs[:min_lookback] = np.log(self.unigram.clip(1e-30)).astype(np.float32)
        
        return log_probs
    
    def save(self, path: Path):
        """Save to disk."""
        data = {
            'V': self.V,
            'unigram': self.unigram,
            'patterns': [(p['probs'], p['backoff'], p['ctx_positions'], p['name']) 
                         for p in self.patterns],
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, path: Path):
        """Load from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls()
        obj.V = data['V']
        obj.unigram = data['unigram']
        obj.patterns = [{'probs': p[0], 'backoff': p[1], 'ctx_positions': p[2], 'name': p[3]}
                        for p in data['patterns']]
        return obj


def main():
    """Build and test SkipGramKNChannel."""
    import sys
    sys.path.insert(0, str(REPO))
    from compile_wiki_lm_v13 import load_setup, load_or_build_tokens
    
    bpe, vocab, tok2id, bpe_to_lm, emb, V_val, d = load_setup()
    ids_all = load_or_build_tokens(bpe, bpe_to_lm, V_val)
    ids_np = ids_all.numpy()
    
    # Build on training data (first 22M tokens)
    train_ids = ids_np[:22_000_000]
    
    print(f"Building SkipGramKNChannel on {len(train_ids):,} tokens...", flush=True)
    sk = SkipGramKNChannel()
    sk.build(train_ids)
    
    # Save
    save_path = ARTIFACT_DIR / "table.pkl"
    sk.save(save_path)
    print(f"Saved to {save_path}", flush=True)
    
    # Test (a): probabilities sum to ≈1
    print("\n=== Test (a): probabilities sum to ≈1 ===", flush=True)
    val_ids = ids_np[22_000_000:22_001_000]
    log_probs = sk.score(val_ids)
    
    for t in [0, 100, 500, 999]:
        probs = np.exp(log_probs[t])
        s = probs.sum()
        print(f"  pos {t}: sum={s:.6f} (should be ~1.0)", flush=True)
    
    # Test (b): PPL on val slice < unigram PPL
    print("\n=== Test (b): PPL < unigram PPL ===", flush=True)
    val_slice = ids_np[22_000_000:22_500_000]
    targets = val_slice[1:]
    
    log_probs_eval = sk.score(val_slice)
    
    # SkipGram PPL
    nll_skip = 0.0
    for t in range(len(targets)):
        nll_skip += -log_probs_eval[t, targets[t]]
    ppl_skip = math.exp(nll_skip / len(targets))
    
    # Unigram PPL
    uni_log = np.log(sk.unigram.clip(1e-30))
    nll_uni = sum(-uni_log[targets[t]] for t in range(len(targets)))
    ppl_uni = math.exp(nll_uni / len(targets))
    
    print(f"  SkipGram KN PPL: {ppl_skip:.1f}", flush=True)
    print(f"  Unigram PPL:      {ppl_uni:.1f}", flush=True)
    print(f"  SkipGram < Uni:   {ppl_skip < ppl_uni}", flush=True)
    
    # Also compute top-1
    correct = sum(1 for t in range(min(10000, len(targets))) 
                  if np.argmax(log_probs_eval[t]) == targets[t])
    print(f"  Top-1 (10K):      {correct/min(10000,len(targets))*100:.2f}%", flush=True)
    
    # Save log_probs to verify shape
    print(f"\n  Log-probs shape: {log_probs.shape}  dtype: {log_probs.dtype}", flush=True)


if __name__ == "__main__":
    main()
