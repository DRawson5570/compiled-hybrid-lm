"""data_filter.py — Real-time prior-based data quality filter (Gemini V4 upgrade).

Uses the fast compiled n-gram prior to evaluate document quality before training.
High PPL → likely garbage/non-English → skip. Low PPL → likely boilerplate/spam → skip.

Filters C4 documents in the data streaming pipeline with zero GPU overhead.
"""
import math
import torch
from collections import defaultdict


class CompiledPriorQualityFilter:
    """Evaluates text quality using compiled n-gram prior PPL.

    Fast, CPU-only, no model forward passes needed.
    """

    def __init__(self, V=50257, high_ppl_threshold=200, low_ppl_threshold=5,
                 min_tokens=32):
        self.V = V
        self.high_ppl = high_ppl_threshold
        self.low_ppl = low_ppl_threshold
        self.min_tokens = min_tokens
        self._uniform = -math.log(V)

        # Lightweight streaming caches
        self._uni = None
        self._bi = {}
        self._bit = {}
        self._ctx = []

    def _ensure_state(self):
        if self._uni is None:
            self._uni = torch.zeros(self.V, dtype=torch.float32)

    def score_document(self, token_ids: torch.Tensor) -> tuple[float, str]:
        """Score a document and return (ppl, verdict).

        Args:
            token_ids: 1D tensor of token IDs for a document

        Returns:
            (perplexity, 'keep'|'high_ppl'|'low_ppl'|'too_short')
        """
        if len(token_ids) < self.min_tokens:
            return 0.0, 'too_short'

        ids = token_ids.long().tolist()
        self._ensure_state()

        total_nll = 0.0
        total_n = 0

        # Reset caches for this document
        self._uni.zero_()
        self._bi.clear(); self._bit.clear()
        self._ctx.clear()

        for i in range(1, len(ids)):
            prev = ids[i - 1]; curr = ids[i]
            self._ctx.append(prev)
            if len(self._ctx) > 128:
                self._ctx = self._ctx[-128:]

            # Decay every 10 steps
            if i % 10 == 0:
                self._uni *= 0.999

            if prev < self.V:
                self._uni[prev] += 1

            # Bigram update
            if prev < self.V and curr < self.V:
                k = (prev, curr)
                self._bi[k] = self._bi.get(k, 0) + 1
                self._bit[prev] = self._bit.get(prev, 0) + 1

            # Compute bigram log-prob of current token
            if prev < self.V and curr < self.V:
                bi_tot = self._bit.get(prev, 0)
                bi_cnt = self._bi.get((prev, curr), 0)
                if bi_tot > 0:
                    p = (bi_cnt + 0.01) / (bi_tot + 0.01 * self.V)
                    total_nll -= math.log(max(p, 1e-12))
                else:
                    total_nll -= self._uniform
                total_n += 1

        if total_n == 0:
            return 0.0, 'too_short'

        ppl = math.exp(total_nll / total_n)

        if ppl > self.high_ppl:
            return ppl, 'high_ppl'
        elif ppl < self.low_ppl:
            return ppl, 'low_ppl'
        return ppl, 'keep'


def filter_corpus(token_ids: torch.Tensor, filter_fn, batch_size=4096,
                  verbose=True) -> torch.Tensor:
    """Filter a corpus using a quality filter, returning clean token IDs.

    Args:
        token_ids: Full corpus tensor
        filter_fn: CompiledPriorQualityFilter instance
        batch_size: Tokens per document chunk (assumes <|endoftext|> separation)

    Returns:
        Filtered token IDs tensor
    """
    # Split on endoftext tokens (GPT-2 EOS = 50256)
    eos_id = 50256
    eos_positions = (token_ids == eos_id).nonzero(as_tuple=True)[0]

    kept = []
    kept_count = 0
    skipped_high = 0
    skipped_low = 0

    start = 0
    for end_pos in eos_positions.tolist():
        doc = token_ids[start:end_pos + 1]
        ppl, verdict = filter_fn.score_document(doc)

        if verdict == 'keep':
            kept.append(doc)
            kept_count += len(doc)
        elif verdict == 'high_ppl':
            skipped_high += 1
        elif verdict == 'low_ppl':
            skipped_low += 1

        start = end_pos + 1

    result = torch.cat(kept) if kept else token_ids[:0]
    if verbose:
        print(f'  Filtered: {len(token_ids):,} → {kept_count:,} tokens '
              f'(skipped {skipped_high} high-PPL, {skipped_low} low-PPL)')
    return result
