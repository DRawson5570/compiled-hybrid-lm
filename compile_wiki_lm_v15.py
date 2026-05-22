"""
compile_wiki_lm_v15.py — Product-of-Experts: v13 mixture × v11 ridge
=====================================================================

Hypothesis: the v13 mixture (per-cluster empirical distributions) and the
v11 ridge head (linear projection of positional concat) capture different
signal:
  - v13 mixture: nonlinear neighbourhood-based distribution lookup;
    sharp on common context patterns, sparse on rare ones.
  - v11 ridge: smooth bigram-style linear features; never sharp but
    nowhere zero.

Their geometric mean (=sum of log-probs, then renormalised) often beats
either alone — this is a classical "Product of Experts" combination
(Hinton 2002), and it's done at inference time with no joint training:

    log P_poe(y|r) = β · log P_mix(y|r) + (1-β) · log P_ridge(y|r) - logZ

β is a single mixture weight, swept on the validation set.

v15 reuses the saved artefacts from v13 (best run) and v11 (best run);
the two LMs are independent and need no re-compilation. The PoE just
sums log-probs with the trained scalar weight and renormalises.

Outputs:
    artifacts/compiled_wiki_lm_v15/eval_results_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from compile_wiki_lm_v13 import (
    load_setup, load_or_build_tokens, build_residual, collect_residuals,
    parse_size, DEVICE,
)

REPO = Path("/home/drawson/llm_decoupling")
ARTIFACT = REPO / "artifacts/compiled_wiki_lm_v15"
ARTIFACT.mkdir(parents=True, exist_ok=True)


class RidgeExpert:
    """v11 ridge LM. Stored as a (V, d_res) matrix W and a scalar T."""

    def __init__(self, W: torch.Tensor, T: float, K_pos: int, V: int, d_emb: int):
        self.W = W.to(DEVICE)
        self.T = T
        self.K_pos = K_pos
        self.V = V
        self.d_emb = d_emb

    def log_probs(self, R: torch.Tensor) -> torch.Tensor:
        logits = R @ self.W.t() * self.T
        return F.log_softmax(logits, dim=-1)


class MixtureExpert:
    """v13 cluster-mixture LM."""

    def __init__(self, mu, log_p_cluster, log_p_uni, K_pos, V, d_emb,
                 tau, gamma):
        self.mu = mu.to(DEVICE)
        self.log_p_cluster = log_p_cluster.to(DEVICE)
        self.log_p_uni = log_p_uni.to(DEVICE)
        self.K_pos = K_pos
        self.V = V
        self.d_emb = d_emb
        self.tau = tau
        self.gamma = gamma
        self._mu_sq = (self.mu * self.mu).sum(dim=1)

    def log_probs(self, R: torch.Tensor, top_M: int = 64) -> torch.Tensor:
        d2 = (R * R).sum(dim=1, keepdim=True) - 2 * R @ self.mu.t() + self._mu_sq[None]
        neg_d2_topM, idx_topM = torch.topk(-d2, k=min(top_M, self.mu.size(0)),
                                            dim=1)
        log_pi = F.log_softmax(neg_d2_topM / self.tau, dim=-1)
        B = R.size(0)
        out = torch.empty(B, self.V, device=R.device, dtype=R.dtype)
        bchunk = max(1, min(B, 256_000_000 // max(top_M * self.V * 4, 1)))
        for s in range(0, B, bchunk):
            e = min(s + bchunk, B)
            lp_sel = self.log_p_cluster[idx_topM[s:e]]
            block = log_pi[s:e].unsqueeze(2) + lp_sel
            out[s:e] = torch.logsumexp(block, dim=1)
        if self.gamma >= 1.0 - 1e-9:
            return out
        log_g = math.log(self.gamma); log_1mg = math.log(1.0 - self.gamma)
        a = log_g + out
        b_ = log_1mg + self.log_p_uni[None].expand_as(out)
        m = torch.maximum(a, b_)
        return m + torch.log(torch.exp(a - m) + torch.exp(b_ - m))


def poe_ppl(R, Y, mix: MixtureExpert, ridge: RidgeExpert | None,
            beta: float, top_M: int = 64, inner_batch: int = 256) -> dict:
    nll_sum = 0.0
    top1 = 0
    top5 = 0
    count = 0
    for i in range(0, R.size(0), inner_batch):
        Rb = R[i:i + inner_batch]
        Yb = Y[i:i + inner_batch]
        log_m = mix.log_probs(Rb, top_M=top_M)
        if ridge is not None and beta < 1.0 - 1e-9:
            log_r = ridge.log_probs(Rb)
            log_p = beta * log_m + (1.0 - beta) * log_r
            log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        else:
            log_p = log_m
        nll_sum += -log_p.gather(1, Yb.unsqueeze(1)).squeeze(1).sum().item()
        top5_idx = log_p.topk(5, dim=-1).indices
        top1 += (top5_idx[:, 0] == Yb).sum().item()
        top5 += (top5_idx == Yb.unsqueeze(1)).any(dim=1).sum().item()
        count += Rb.size(0)
    avg = nll_sum / count
    return {"count": count, "ppl": math.exp(avg), "avg_nll": avg,
            "top1": top1 / count, "top5": top5 / count, "beta": beta}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v13-path", type=str,
                   default=str(REPO / "artifacts/compiled_wiki_lm_v13/compiled_lm_scale1.pt"))
    p.add_argument("--v11-path", type=str,
                   default=str(REPO / "artifacts/compiled_wiki_lm_v11/compiled_lm_k3_calib.pt"))
    p.add_argument("--val-tokens", type=str, default="100K")
    p.add_argument("--eval-tokens", type=str, default="300K")
    p.add_argument("--train-tokens", type=str, default="10M")
    p.add_argument("--chunk", type=str, default="80K")
    p.add_argument("--inner-batch", type=int, default=256)
    p.add_argument("--top-M", type=int, default=64)
    p.add_argument("--tag", type=str, default="default")
    args = p.parse_args()

    train_n = parse_size(args.train_tokens)
    val_n = parse_size(args.val_tokens)
    eval_n = parse_size(args.eval_tokens)
    chunk = parse_size(args.chunk)

    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    ids = load_or_build_tokens(bpe, bpe_to_lm, V)
    N = ids.size(0)
    val_ids = ids[train_n:train_n + val_n]
    eval_ids = ids[-eval_n:]

    # Load v13
    v13 = torch.load(args.v13_path, map_location="cpu", weights_only=False)
    K_pos = v13["K_pos"]
    print(f"[v13] {args.v13_path} K_pos={K_pos} clusters={v13['clusters']} "
          f"τ={v13['best_tau']} γ={v13['best_gamma']}")
    mix = MixtureExpert(
        mu=v13["mu"], log_p_cluster=v13["log_p_cluster"],
        log_p_uni=v13["log_p_uni"], K_pos=K_pos, V=V, d_emb=d,
        tau=v13["best_tau"], gamma=v13["best_gamma"],
    )

    # Load v11
    v11_path = Path(args.v11_path)
    v11 = torch.load(args.v11_path, map_location="cpu", weights_only=False)
    assert v11["K"] == K_pos, f"K mismatch: v13={K_pos}, v11={v11['K']}"
    # v11 saved W with `use_bias=True` so residual has trailing constant-1
    # channel. v13's residual layout omits the bias. Strip bias column from W
    # so the same residual works for both experts.
    W = v11["W"].float()
    d_res_v11 = W.size(1)
    d_res_v13 = (K_pos + 1) * d
    if d_res_v11 == d_res_v13 + 1:
        print(f"[v11] stripping bias column ({d_res_v11} -> {d_res_v13})")
        W = W[:, :d_res_v13]
    elif d_res_v11 == d_res_v13:
        pass
    else:
        raise ValueError(f"v11 d_res={d_res_v11} incompatible with v13 d_res={d_res_v13}")
    # Look up calibrated T from v11 eval json if available
    # (saved file does not include T; we hardcode 50 — the known optimum)
    T_v11 = 50.0
    ridge = RidgeExpert(W=W, T=T_v11, K_pos=K_pos, V=V, d_emb=d)
    print(f"[v11] W={tuple(W.shape)} T={T_v11}")

    emb_dev = emb.to(DEVICE)
    print(f"\n[cal] precomputing residuals on {val_n:,} val tokens")
    cal_R, cal_Y = collect_residuals(val_ids, emb_dev, K_pos, chunk=chunk)
    # cal residuals from v13 builder have NO bias channel (matches both
    # after the bias strip above).
    print(f"  R={tuple(cal_R.shape)}")

    # Baseline checks
    r_mix = poe_ppl(cal_R, cal_Y, mix, None, beta=1.0,
                    top_M=args.top_M, inner_batch=args.inner_batch)
    print(f"  v13 mix alone: PPL={r_mix['ppl']:.2f}")
    r_ridge = poe_ppl(cal_R, cal_Y, mix, ridge, beta=0.0,
                       top_M=args.top_M, inner_batch=args.inner_batch)
    print(f"  v11 ridge alone: PPL={r_ridge['ppl']:.2f}")

    # PoE β sweep
    best = {"ppl": float("inf"), "beta": 1.0}
    for beta in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]:
        r = poe_ppl(cal_R, cal_Y, mix, ridge, beta=beta,
                    top_M=args.top_M, inner_batch=args.inner_batch)
        print(f"  β={beta} → PPL={r['ppl']:.2f}  top1={r['top1']*100:.2f}%")
        if r["ppl"] < best["ppl"]:
            best = {"ppl": r["ppl"], "beta": beta}
    print(f"[cal] best β={best['beta']} PPL={best['ppl']:.2f}")
    del cal_R, cal_Y

    # Final heldout eval
    R_h, Y_h = collect_residuals(eval_ids, emb_dev, K_pos, chunk=chunk)
    held = poe_ppl(R_h, Y_h, mix, ridge, beta=best["beta"],
                   top_M=args.top_M, inner_batch=args.inner_batch)
    held_mix = poe_ppl(R_h, Y_h, mix, None, beta=1.0,
                       top_M=args.top_M, inner_batch=args.inner_batch)
    held_ridge = poe_ppl(R_h, Y_h, mix, ridge, beta=0.0,
                          top_M=args.top_M, inner_batch=args.inner_batch)
    print(f"\n[heldout] v13 mix alone : PPL={held_mix['ppl']:.2f}")
    print(f"[heldout] v11 ridge alone: PPL={held_ridge['ppl']:.2f}")
    print(f"[heldout] PoE β={best['beta']} : PPL={held['ppl']:.2f}  "
          f"top1={held['top1']*100:.2f}%  top5={held['top5']*100:.2f}%")

    results = {
        "model": "Compiled Wikitext LM v15 (Product of Experts: v13 ⊗ v11)",
        "v13_path": str(args.v13_path),
        "v11_path": str(args.v11_path),
        "K_pos": K_pos, "V": V, "d_emb": d,
        "best_beta": best["beta"],
        "top_M": args.top_M,
        "eval_tokens": eval_n,
        "heldout_mix_alone": held_mix,
        "heldout_ridge_alone": held_ridge,
        "heldout_poe": held,
    }
    rp = ARTIFACT / f"eval_results_{args.tag}.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
