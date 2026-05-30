"""hybrid/v1_blender/add_nn_channel.py

Augment a dumped feature npz with a 13th channel from the trained tiny
transformer LM.  Adds (T, 1) slices for log_p_targets/log_p_observed/
log_p_lag1/entropy/max_log_prob/top1_id and a (T, 1, K) slice for
topk_log_probs, then writes a new npz next to the input.

The NN channel is causal by construction (causal mask + position-wise softmax)
so per-position log-probs at index `i` use only ids[:i+1], matching the
no-leak contract of the existing channels.

Usage:
    python hybrid/v1_blender/add_nn_channel.py \\
        --in-npz hybrid/v1_blender/data_big/val.npz \\
        --out-npz hybrid/v1_blender/data_big_nn/val.npz \\
        --cache artifacts/compiled_wiki_lm_v11/cache_lm_ids.pt \\
        --slice-start 22000000 --slice-len 500000 \\
        --K-pos 2 \\
        --nn-ckpt artifacts/nn_channel/nn_channel_v1.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.v1_blender.train_nn_channel import TinyTransformerLM


@torch.no_grad()
def compute_nn_log_probs(model: TinyTransformerLM, ids: torch.Tensor,
                         ctx: int, device: torch.device,
                         batch: int = 16) -> torch.Tensor:
    """Compute (N-1, V) log-probs for predicting ids[i+1] from ids[:i+1].

    Sliding-window scheme:
      - window length = ctx (model's max input)
      - stride        = ctx - 1
      - window k covers input tokens ids[s_k : s_k + ctx]  (where s_k = k*(ctx-1))
      - it yields next-token log-probs for positions s_k+1 .. s_k+ctx-1
        (i.e. ctx-1 predictions, one per position)

    The first window's first position predicts ids[1] from ids[:1]; later
    windows always have ctx tokens of left context, so per-position context
    is always at most ctx tokens, never more.

    Tail handling: the final window may be shorter than ctx; we still pass
    it through and read off whatever positions it covers.
    """
    model.eval()
    N = ids.shape[0]
    V = model.vocab
    out = torch.empty(N - 1, V, dtype=torch.float32)

    stride = ctx - 1
    # Window starts.  s=0 covers preds for positions 1..ctx-1.
    # s=stride covers preds for positions stride+1..stride+ctx-1, etc.
    starts = list(range(0, N - 1, stride))

    pbar_every = max(1, len(starts) // 50)
    import time as _t
    t0 = _t.time()
    for bi in range(0, len(starts), batch):
        bs = starts[bi:bi + batch]
        # Build per-row inputs of variable length (last batch may have a
        # short tail window).  Run them grouped by length where possible.
        for s in bs:
            in_end = min(s + ctx, N)
            in_ids = ids[s:in_end].unsqueeze(0).to(device)  # (1, L)
            L = in_ids.shape[1]
            assert L <= ctx, (L, ctx)
            logits = model(in_ids)  # (1, L, V)
            # logits[0, i] predicts token at position s + i + 1 (if it exists).
            # Valid prediction range: i in [0, L-2] -> positions s+1 .. s+L-1.
            n_preds = L - 1
            if n_preds <= 0:
                continue
            lp = F.log_softmax(logits[0, :n_preds], dim=-1).cpu()
            # Out indexing: position p is stored at out[p - 1].
            out[s : s + n_preds] = lp
        if (bi // batch) % pbar_every == 0:
            done = bi + len(bs)
            pct = 100.0 * done / max(1, len(starts))
            elapsed = _t.time() - t0
            print(f"  nn channel: {done}/{len(starts)} windows ({pct:.1f}%)  {elapsed:.1f}s")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-npz", required=True)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--cache", default="artifacts/compiled_wiki_lm_v11/cache_lm_ids.pt")
    p.add_argument("--slice-start", type=int, required=True,
                   help="Token index where this slice starts in the cache.")
    p.add_argument("--slice-len", type=int, required=True,
                   help="Length of the slice (in tokens) before K_pos drop.")
    p.add_argument("--K-pos", type=int, default=2)
    p.add_argument("--nn-ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[load] npz {args.in_npz}")
    in_npz = np.load(args.in_npz, allow_pickle=True)
    T = in_npz["log_p_targets"].shape[0]
    C_old = in_npz["log_p_targets"].shape[1]
    print(f"  T={T:,}  C_old={C_old}")

    print(f"[load] tokens {args.cache}")
    ids = torch.load(args.cache).long()
    sl = ids[args.slice_start : args.slice_start + args.slice_len]
    assert sl.shape[0] == args.slice_len, (sl.shape[0], args.slice_len)
    # Channel arrays index the next-token at position K_pos + i (target),
    # observed at K_pos + i - 1 (current).  The slice we feed the NN is the
    # full (slice_len) window so all references stay correct.

    print(f"[load] nn ckpt {args.nn_ckpt}")
    ck = torch.load(args.nn_ckpt, map_location=device)
    cfg = ck["config"]
    model = TinyTransformerLM(
        vocab=cfg["vocab"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"], ctx=cfg["ctx"], n_layers=cfg["n_layers"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"  d_model={cfg['d_model']} n_layers={cfg['n_layers']} ctx={cfg['ctx']} V={cfg['vocab']}")

    # Compute (slice_len-1, V) log-probs for predicting sl[i] from sl[:i].
    sl_dev = sl.to(device)
    print(f"[nn] computing log-probs over {sl.shape[0]:,} tokens (ctx={cfg['ctx']})")
    lp_full = compute_nn_log_probs(model, sl_dev, ctx=cfg["ctx"], device=device)
    # lp_full[i] = log p(sl[i+1] context = sl[:i+1])
    print(f"  nn log-probs shape={tuple(lp_full.shape)}")

    # The existing channels at array index `t` correspond to:
    #   target  = ids_n[K_pos + t]   (slice index K_pos + t)
    #   observed = ids_n[K_pos + t - 1]
    # The NN log-probs lp_full[i] predicts sl[i+1] from sl[:i+1].
    # So lp_full[K_pos + t - 1] predicts sl[K_pos + t] given sl[:K_pos + t]  ✓
    K = args.K_pos
    expected_T = sl.shape[0] - K
    assert expected_T == T, (expected_T, T)
    nn_lp = lp_full[K - 1 : K - 1 + T]  # (T, V)
    assert nn_lp.shape[0] == T, (nn_lp.shape, T)
    V = nn_lp.shape[1]

    # Compute per-channel summary stats matching dump_features.summarize signature
    targets = torch.from_numpy(in_npz["targets"].astype(np.int64))
    observed = torch.from_numpy(in_npz["observed"].astype(np.int64))
    # lag1: same shifted-down-by-1 pattern as dump_features.summarize
    lag1 = torch.cat([observed[:1], observed[:-1]])

    rows = torch.arange(T)
    log_p_targets = nn_lp[rows, targets].numpy().reshape(T, 1).astype(np.float32)
    log_p_observed = nn_lp[rows, observed].numpy().reshape(T, 1).astype(np.float32)
    log_p_lag1 = nn_lp[rows, lag1].numpy().reshape(T, 1).astype(np.float32)

    # entropy / max / top-K — chunked for memory
    entropy = np.zeros((T, 1), dtype=np.float32)
    max_log_prob = np.zeros((T, 1), dtype=np.float32)
    top1_id = np.zeros((T, 1), dtype=np.int32)
    topk_log_probs = np.zeros((T, 1, args.top_k), dtype=np.float32)
    chunk = 4096
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        lp = nn_lp[s:e].numpy()  # (b, V)
        p = np.exp(lp)
        entropy[s:e, 0] = -(p * lp).sum(axis=1)
        # top-k
        part_idx = np.argpartition(-lp, args.top_k, axis=1)[:, :args.top_k]
        part_vals = np.take_along_axis(lp, part_idx, axis=1)
        order = np.argsort(-part_vals, axis=1)
        topk_sorted = np.take_along_axis(part_vals, order, axis=1)
        ids_sorted = np.take_along_axis(part_idx, order, axis=1)
        topk_log_probs[s:e, 0, :] = topk_sorted
        max_log_prob[s:e, 0] = topk_sorted[:, 0]
        top1_id[s:e, 0] = ids_sorted[:, 0].astype(np.int32)

    # Concatenate as new channel
    out = {}
    for k in in_npz.files:
        out[k] = in_npz[k]
    out["log_p_targets"] = np.concatenate([in_npz["log_p_targets"], log_p_targets], axis=1)
    out["log_p_observed"] = np.concatenate([in_npz["log_p_observed"], log_p_observed], axis=1)
    out["log_p_lag1"] = np.concatenate([in_npz["log_p_lag1"], log_p_lag1], axis=1)
    out["entropy"] = np.concatenate([in_npz["entropy"], entropy], axis=1)
    out["max_log_prob"] = np.concatenate([in_npz["max_log_prob"], max_log_prob], axis=1)
    out["top1_id"] = np.concatenate([in_npz["top1_id"], top1_id], axis=1)
    if "topk_log_probs" in in_npz.files:
        old_top = in_npz["topk_log_probs"]  # (T, C, K_old)
        K_old = old_top.shape[2]
        if K_old != args.top_k:
            print(f"  resizing topk: existing K={K_old} new K={args.top_k}")
            # Reshape NN topk to match existing K (truncate or zero-pad)
            if args.top_k > K_old:
                topk_log_probs = topk_log_probs[:, :, :K_old]
            else:
                pad = np.full((T, 1, K_old - args.top_k), fill_value=-1e9, dtype=np.float32)
                topk_log_probs = np.concatenate([topk_log_probs, pad], axis=2)
        out["topk_log_probs"] = np.concatenate([old_top, topk_log_probs], axis=1)
    out["channel_names"] = np.concatenate([in_npz["channel_names"], np.array(["nn"])])

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **out)
    print(f"[save] {args.out_npz}")
    print(f"  C_new={out['log_p_targets'].shape[1]}  channels: {out['channel_names'].tolist()}")

    # Quick sanity: per-channel ppl on this slice
    import math
    print("\n[ref] per-channel PPL on this slice:")
    for ci, cname in enumerate(out["channel_names"]):
        nll = -out["log_p_targets"][:, ci].mean()
        print(f"  ch{ci:2d} {cname:8s}  ppl={math.exp(nll):8.2f}")


if __name__ == "__main__":
    main()
