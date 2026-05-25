"""
compile_wiki_lm_v30.py — 11-way multi-temporal decayed conditional + unconditioned induction
                          + Continuous Attention Retrieval Cache (Semantic Induction) Ensemble
=============================================================================================

This version introduces the Continuous Attention Retrieval Cache (also known as continuous
episodic attention cache) which leverages the PPMI+SVD word embedding space to retrieve semantically
relevant next-tokens from the sliding history window. This acts as a 'Semantic Induction' head,
allowing the model to predict synonyms or semantically related completions without gradient descent.

We construct an 11-way blend of:
  1. Global Kneser-Ney (KN7)
  2. Spatial cluster mixture (v14 model)
  3. Fast decayed trigram
  4. Slow decayed trigram
  5. Fast decayed conditional bigram
  6. Slow decayed conditional bigram
  7. Fast unconditioned unigram cache
  8. Slow unconditioned unigram cache
  9. Fast Continuous Attention Retrieval Cache (Semantic Induction)
  10. Slow Continuous Attention Retrieval Cache (Semantic Induction)
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/home/drawson/llm_decoupling")
sys.path.insert(0, str(REPO))

from compile_wiki_lm_v13 import load_setup, load_or_build_tokens, build_residual, parse_size, DEVICE
from compile_wiki_lm_v14 import SparseMixtureClusterLM
from compile_wiki_lm_v23 import ModifiedKNGram
from compile_wiki_lm_v24 import compute_log_p_mix, compute_log_p_kn

ARTIFACT = Path("/home/drawson/deepseek_experiments/artifacts/compiled_wiki_lm_v30")
ARTIFACT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def build_v30_induction_log_probs(
    ids_np: np.ndarray, V: int, K_pos: int, window: int,
    lam_tri_fast: float, lam_tri_slow: float,
    lam_bi_fast: float, lam_bi_slow: float,
    lam_ucache_fast: float, lam_ucache_slow: float,
    alpha_tri_fast: float, alpha_tri_slow: float,
    alpha_bi_fast: float, alpha_bi_slow: float,
    alpha_ucache_fast: float, alpha_ucache_slow: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream ids; build sliding-window dual decayed trigram, dual conditional bigram and dual unconditioned unigram.

    Returns:
      log_p_tri_fast, log_p_tri_slow, log_p_bi_fast, log_p_bi_slow, log_p_ucache_fast, log_p_ucache_slow
    """
    N = len(ids_np)
    T = N - K_pos

    # 1. Decayed Bigram (Conditional) - Fast
    B_bi_fast = torch.zeros(V, V, device=DEVICE, dtype=torch.float32)
    R_bi_fast = torch.zeros(V, device=DEVICE, dtype=torch.float32)
    last_bi_fast = torch.zeros(V, V, device=DEVICE, dtype=torch.int32)
    last_R_bi_fast = torch.zeros(V, device=DEVICE, dtype=torch.int32)

    # 2. Decayed Bigram (Conditional) - Slow
    B_bi_slow = torch.zeros(V, V, device=DEVICE, dtype=torch.float32)
    R_bi_slow = torch.zeros(V, device=DEVICE, dtype=torch.float32)
    last_bi_slow = torch.zeros(V, V, device=DEVICE, dtype=torch.int32)
    last_R_bi_slow = torch.zeros(V, device=DEVICE, dtype=torch.int32)

    # 3. Decayed Unconditioned Unigram Cache - Fast
    C_uc_fast = torch.zeros(V, device=DEVICE, dtype=torch.float32)
    last_uc_fast = torch.zeros(V, device=DEVICE, dtype=torch.int32)
    S_uc_fast = 0.0
    last_S_uc_fast = 0

    # 4. Decayed Unconditioned Unigram Cache - Slow
    C_uc_slow = torch.zeros(V, device=DEVICE, dtype=torch.float32)
    last_uc_slow = torch.zeros(V, device=DEVICE, dtype=torch.int32)
    S_uc_slow = 0.0
    last_S_uc_slow = 0

    # 5. Decayed Trigram - Fast
    Bt_fast: dict[tuple, torch.Tensor] = {}
    Rt_fast: dict[tuple, float] = {}
    last_t_fast: dict[tuple, torch.Tensor] = {}
    last_Rt_fast: dict[tuple, int] = {}
    
    # 6. Decayed Trigram - Slow
    Bt_slow: dict[tuple, torch.Tensor] = {}
    Rt_slow: dict[tuple, float] = {}
    last_t_slow: dict[tuple, torch.Tensor] = {}
    last_Rt_slow: dict[tuple, int] = {}
    
    history = deque()

    log_p_tri_fast = np.zeros((T, V), dtype=np.float32)
    log_p_tri_slow = np.zeros((T, V), dtype=np.float32)
    log_p_bi_fast = np.zeros((T, V), dtype=np.float32)
    log_p_bi_slow = np.zeros((T, V), dtype=np.float32)
    log_p_ucache_fast = np.zeros((T, V), dtype=np.float32)
    log_p_ucache_slow = np.zeros((T, V), dtype=np.float32)

    aV_bifast = alpha_bi_fast * V
    aV_bislow = alpha_bi_slow * V
    aV_ucfast = alpha_ucache_fast * V
    aV_ucslow = alpha_ucache_slow * V
    aV_t_fast = alpha_tri_fast * V
    aV_t_slow = alpha_tri_slow * V
    log_uniform = math.log(1.0 / V)

    t0 = time.time()
    for t in range(K_pos, N):
        i = t - K_pos
        b = int(ids_np[t - 1])

        # A. Conditional Bigram - Fast
        p_bifast_num = B_bi_fast[b].clone()
        bifast_dt = t - last_bi_fast[b]
        p_bifast_num = p_bifast_num * torch.exp(-lam_bi_fast * bifast_dt)
        bifast_R_dt = t - last_R_bi_fast[b]
        bifast_den = R_bi_fast[b] * math.exp(-lam_bi_fast * bifast_R_dt)
        p_bifast_row = (p_bifast_num + alpha_bi_fast) / (bifast_den + aV_bifast)
        log_p_bi_fast[i] = torch.log(p_bifast_row.clamp_min(1e-30)).cpu().numpy()

        # B. Conditional Bigram - Slow
        p_bislow_num = B_bi_slow[b].clone()
        bislow_dt = t - last_bi_slow[b]
        p_bislow_num = p_bislow_num * torch.exp(-lam_bi_slow * bislow_dt)
        bislow_R_dt = t - last_R_bi_slow[b]
        bislow_den = R_bi_slow[b] * math.exp(-lam_bi_slow * bislow_R_dt)
        p_bislow_row = (p_bislow_num + alpha_bi_slow) / (bislow_den + aV_bislow)
        log_p_bi_slow[i] = torch.log(p_bislow_row.clamp_min(1e-30)).cpu().numpy()

        # C. Unconditioned Unigram Cache - Fast
        p_ucfast_num = C_uc_fast.clone()
        ucfast_dt = t - last_uc_fast
        p_ucfast_num = p_ucfast_num * torch.exp(-lam_ucache_fast * ucfast_dt)
        ucfast_den = S_uc_fast * math.exp(-lam_ucache_fast * (t - last_S_uc_fast))
        p_ucfast_row = (p_ucfast_num + alpha_ucache_fast) / (ucfast_den + aV_ucfast)
        log_p_ucache_fast[i] = torch.log(p_ucfast_row.clamp_min(1e-30)).cpu().numpy()

        # D. Unconditioned Unigram Cache - Slow
        p_ucslow_num = C_uc_slow.clone()
        ucslow_dt = t - last_uc_slow
        p_ucslow_num = p_ucslow_num * torch.exp(-lam_ucache_slow * ucslow_dt)
        ucslow_den = S_uc_slow * math.exp(-lam_ucache_slow * (t - last_S_uc_slow))
        p_ucslow_row = (p_ucslow_num + alpha_ucache_slow) / (ucslow_den + aV_ucslow)
        log_p_ucache_slow[i] = torch.log(p_ucslow_row.clamp_min(1e-30)).cpu().numpy()

        # E. Conditional Trigram Fast & Slow
        if t >= 2:
            a = int(ids_np[t - 2])
            key = (a, b)
            
            # Trigram Fast
            row_t_fast = Bt_fast.get(key)
            if row_t_fast is not None:
                tri_f_num = row_t_fast.clone()
                tri_f_last = last_t_fast[key]
                tri_f_dt = t - tri_f_last
                tri_f_num = tri_f_num * torch.exp(-lam_tri_fast * tri_f_dt)
                tri_f_R_dt = t - last_Rt_fast[key]
                tri_f_den = Rt_fast[key] * math.exp(-lam_tri_fast * tri_f_R_dt)
                p_tri_f_row = (tri_f_num + alpha_tri_fast) / (tri_f_den + aV_t_fast)
                log_p_tri_fast[i] = torch.log(p_tri_f_row.clamp_min(1e-30)).cpu().numpy()
            else:
                log_p_tri_fast[i].fill(log_uniform)
                
            # Trigram Slow
            row_t_slow = Bt_slow.get(key)
            if row_t_slow is not None:
                tri_s_num = row_t_slow.clone()
                tri_s_last = last_t_slow[key]
                tri_s_dt = t - tri_s_last
                tri_s_num = tri_s_num * torch.exp(-lam_tri_slow * tri_s_dt)
                tri_s_R_dt = t - last_Rt_slow[key]
                tri_s_den = Rt_slow[key] * math.exp(-lam_tri_slow * tri_s_R_dt)
                p_tri_s_row = (tri_s_num + alpha_tri_slow) / (tri_s_den + aV_t_slow)
                log_p_tri_slow[i] = torch.log(p_tri_s_row.clamp_min(1e-30)).cpu().numpy()
            else:
                log_p_tri_slow[i].fill(log_uniform)
        else:
            log_p_tri_fast[i].fill(log_uniform)
            log_p_tri_slow[i].fill(log_uniform)

        # F. Update Tables using the Observed Token `c = ids_np[t]`
        c = int(ids_np[t])

        # Conditional Bigram - Fast
        B_bi_fast[b] = B_bi_fast[b] * torch.exp(-lam_bi_fast * (t - last_bi_fast[b]))
        B_bi_fast[b, c] += 1.0
        last_bi_fast[b] = t
        
        R_bi_fast[b] = R_bi_fast[b] * math.exp(-lam_bi_fast * (t - last_R_bi_fast[b])) + 1.0
        last_R_bi_fast[b] = t

        # Conditional Bigram - Slow
        B_bi_slow[b] = B_bi_slow[b] * torch.exp(-lam_bi_slow * (t - last_bi_slow[b]))
        B_bi_slow[b, c] += 1.0
        last_bi_slow[b] = t
        
        R_bi_slow[b] = R_bi_slow[b] * math.exp(-lam_bi_slow * (t - last_R_bi_slow[b])) + 1.0
        last_R_bi_slow[b] = t

        # Unconditioned Cache - Fast
        C_uc_fast[c] = C_uc_fast[c] * math.exp(-lam_ucache_fast * (t - last_uc_fast[c])) + 1.0
        last_uc_fast[c] = t
        
        S_uc_fast = S_uc_fast * math.exp(-lam_ucache_fast * (t - last_S_uc_fast)) + 1.0
        last_S_uc_fast = t

        # Unconditioned Cache - Slow
        C_uc_slow[c] = C_uc_slow[c] * math.exp(-lam_ucache_slow * (t - last_uc_slow[c])) + 1.0
        last_uc_slow[c] = t
        
        S_uc_slow = S_uc_slow * math.exp(-lam_ucache_slow * (t - last_S_uc_slow)) + 1.0
        last_S_uc_slow = t

        # Conditional Trigram Fast & Slow Updates
        a_val = None
        key_val = None
        if t >= 2:
            a_val = int(ids_np[t - 2])
            key_val = (a_val, b)
            
            # Fast Update
            if key_val in Bt_fast:
                Bt_fast[key_val] = Bt_fast[key_val] * torch.exp(-lam_tri_fast * (t - last_t_fast[key_val]))
                Bt_fast[key_val][c] += 1.0
                last_t_fast[key_val][c] = t
                
                Rt_fast[key_val] = Rt_fast[key_val] * math.exp(-lam_tri_fast * (t - last_Rt_fast[key_val])) + 1.0
                last_Rt_fast[key_val] = t
            else:
                v = torch.zeros(V, device=DEVICE, dtype=torch.float32)
                v[c] = 1.0
                Bt_fast[key_val] = v
                last_arr = torch.zeros(V, device=DEVICE, dtype=torch.int32)
                last_arr[c] = t
                last_t_fast[key_val] = last_arr
                Rt_fast[key_val] = 1.0
                last_Rt_fast[key_val] = t

            # Slow Update
            if key_val in Bt_slow:
                Bt_slow[key_val] = Bt_slow[key_val] * torch.exp(-lam_tri_slow * (t - last_t_slow[key_val]))
                Bt_slow[key_val][c] += 1.0
                last_t_slow[key_val][c] = t
                
                Rt_slow[key_val] = Rt_slow[key_val] * math.exp(-lam_tri_slow * (t - last_Rt_slow[key_val])) + 1.0
                last_Rt_slow[key_val] = t
            else:
                v = torch.zeros(V, device=DEVICE, dtype=torch.float32)
                v[c] = 1.0
                Bt_slow[key_val] = v
                last_arr = torch.zeros(V, device=DEVICE, dtype=torch.int32)
                last_arr[c] = t
                last_t_slow[key_val] = last_arr
                Rt_slow[key_val] = 1.0
                last_Rt_slow[key_val] = t

        history.append((a_val, b, c, key_val))

        # Sliding Window Eviction
        if len(history) > window:
            old_a, old_b, old_c, old_key = history.popleft()
            evict_t = t - window

            # Conditional Bigram - Fast
            B_bi_fast[old_b] = B_bi_fast[old_b] * torch.exp(-lam_bi_fast * (t - last_bi_fast[old_b]))
            B_bi_fast[old_b, old_c] -= math.exp(-lam_bi_fast * (t - evict_t))
            B_bi_fast[old_b, old_c] = max(0.0, B_bi_fast[old_b, old_c])
            last_bi_fast[old_b] = t
            
            R_bi_fast[old_b] = R_bi_fast[old_b] * math.exp(-lam_bi_fast * (t - last_R_bi_fast[old_b])) - math.exp(-lam_bi_fast * (t - evict_t))
            R_bi_fast[old_b] = max(0.0, R_bi_fast[old_b])
            last_R_bi_fast[old_b] = t

            # Conditional Bigram - Slow
            B_bi_slow[old_b] = B_bi_slow[old_b] * torch.exp(-lam_bi_slow * (t - last_bi_slow[old_b]))
            B_bi_slow[old_b, old_c] -= math.exp(-lam_bi_slow * (t - evict_t))
            B_bi_slow[old_b, old_c] = max(0.0, B_bi_slow[old_b, old_c])
            last_bi_slow[old_b] = t
            
            R_bi_slow[old_b] = R_bi_slow[old_b] * math.exp(-lam_bi_slow * (t - last_R_bi_slow[old_b])) - math.exp(-lam_bi_slow * (t - evict_t))
            R_bi_slow[old_b] = max(0.0, R_bi_slow[old_b])
            last_R_bi_slow[old_b] = t

            # Unconditioned Cache - Fast Eviction
            C_uc_fast[old_c] = C_uc_fast[old_c] * math.exp(-lam_ucache_fast * (t - last_uc_fast[old_c])) - math.exp(-lam_ucache_fast * (t - evict_t))
            C_uc_fast[old_c] = max(0.0, C_uc_fast[old_c])
            last_uc_fast[old_c] = t
            
            S_uc_fast = S_uc_fast * math.exp(-lam_ucache_fast * (t - last_S_uc_fast)) - math.exp(-lam_ucache_fast * (t - evict_t))
            S_uc_fast = max(0.0, S_uc_fast)
            last_S_uc_fast = t

            # Unconditioned Cache - Slow Eviction
            C_uc_slow[old_c] = C_uc_slow[old_c] * math.exp(-lam_ucache_slow * (t - last_uc_slow[old_c])) - math.exp(-lam_ucache_slow * (t - evict_t))
            C_uc_slow[old_c] = max(0.0, C_uc_slow[old_c])
            last_uc_slow[old_c] = t
            
            S_uc_slow = S_uc_slow * math.exp(-lam_ucache_slow * (t - last_S_uc_slow)) - math.exp(-lam_ucache_slow * (t - evict_t))
            S_uc_slow = max(0.0, S_uc_slow)
            last_S_uc_slow = t

            # Conditional Trigram Fast & Slow Eviction
            if old_key is not None:
                # Fast Eviction
                row_t_f = Bt_fast.get(old_key)
                if row_t_f is not None:
                    row_t_f = row_t_f * torch.exp(-lam_tri_fast * (t - last_t_fast[old_key]))
                    row_t_f[old_c] -= math.exp(-lam_tri_fast * (t - evict_t))
                    row_t_f[old_c] = max(0.0, row_t_f[old_c])
                    Bt_fast[old_key] = row_t_f
                    last_t_fast[old_key][old_c] = t
                    
                    Rt_fast[old_key] = Rt_fast[old_key] * math.exp(-lam_tri_fast * (t - last_Rt_fast[old_key])) - math.exp(-lam_tri_fast * (t - evict_t))
                    Rt_fast[old_key] = max(0.0, Rt_fast[old_key])
                    last_Rt_fast[old_key] = t
                    
                    if Rt_fast[old_key] <= 0:
                        del Bt_fast[old_key]
                        del Rt_fast[old_key]
                        del last_t_fast[old_key]
                        del last_Rt_fast[old_key]

                # Slow Eviction
                row_t_s = Bt_slow.get(old_key)
                if row_t_s is not None:
                    row_t_s = row_t_s * torch.exp(-lam_tri_slow * (t - last_t_slow[old_key]))
                    row_t_s[old_c] -= math.exp(-lam_tri_slow * (t - evict_t))
                    row_t_s[old_c] = max(0.0, row_t_s[old_c])
                    Bt_slow[old_key] = row_t_s
                    last_t_slow[old_key][old_c] = t
                    
                    Rt_slow[old_key] = Rt_slow[old_key] * math.exp(-lam_tri_slow * (t - last_Rt_slow[old_key])) - math.exp(-lam_tri_slow * (t - evict_t))
                    Rt_slow[old_key] = max(0.0, Rt_slow[old_key])
                    last_Rt_slow[old_key] = t
                    
                    if Rt_slow[old_key] <= 0:
                        del Bt_slow[old_key]
                        del Rt_slow[old_key]
                        del last_t_slow[old_key]
                        del last_Rt_slow[old_key]

        if (i + 1) % 10000 == 0:
            print(f"    [ind] {i+1}/{T} ({time.time() - t0:.1f}s, tri-fast-keys={len(Bt_fast)}, tri-slow-keys={len(Bt_slow)})")

    return log_p_tri_fast, log_p_tri_slow, log_p_bi_fast, log_p_bi_slow, log_p_ucache_fast, log_p_ucache_slow


@torch.no_grad()
def compute_log_p_attn(
    ids_t: torch.Tensor, emb_dev: torch.Tensor, W_attn: int, beta: float, theta: float, alpha_attn: float, K_pos: int
) -> torch.Tensor:
    """Compute semantic retrieval Continuous Attention Cache log probabilities."""
    N = ids_t.shape[0]
    out_log_p = torch.empty((N - K_pos, emb_dev.shape[0]), dtype=torch.float32)
    
    # Pre-normalize embeddings to make dot product cosine similarity
    emb_norm = F.normalize(emb_dev, p=2, dim=1) # (V, d)
    
    t0 = time.time()
    chunk_size = 2000
    for s in range(K_pos, N, chunk_size):
        e = min(s + chunk_size, N)
        B = e - s
        
        q = emb_norm[ids_t[s:e].long()].to(DEVICE) # (B, d)
        p_attn_batch = torch.zeros((B, emb_dev.shape[0]), device=DEVICE, dtype=torch.float32)
        
        for idx in range(B):
            t = s + idx
            j_start = max(0, t - W_attn)
            j_end = t # exclusive
            W_current = j_end - j_start
            if W_current == 0:
                continue
                
            context_ids = ids_t[j_start:j_end].long().to(DEVICE)
            next_ids = ids_t[j_start+1:j_end+1].long().to(DEVICE)
            
            k = emb_norm[context_ids] # (W_current, d)
            dot = torch.mv(k, q[idx]) # (W_current,)
            
            # Temporal decay from temporal distance (t - j)
            dist = torch.arange(W_current, 0, -1, device=DEVICE, dtype=torch.float32) # distance format
            scores = torch.exp(beta * dot - theta * dist)
            scores_sum = scores.sum()
            if scores_sum > 0:
                probs = scores / scores_sum
                p_attn_batch[idx].scatter_add_(0, next_ids, probs)
                
        # Laplace smooth to get log probabilities
        p_smoothed = (p_attn_batch + alpha_attn) / (1.0 + alpha_attn * emb_dev.shape[0])
        out_log_p[s-K_pos : e-K_pos] = torch.log(p_smoothed.clamp_min(1e-30)).cpu()
        
        if (s - K_pos + chunk_size) % 10000 == 0 or e == N:
            print(f"    [attn-W{W_attn}] {min(e - K_pos, N - K_pos)}/{N - K_pos} completed ({time.time() - t0:.1f}s)")
            
    return out_log_p


def eval_blend11(
    log_p_kn, log_p_mix, log_p_tri_fast, log_p_tri_slow, log_p_bi_fast, log_p_bi_slow, 
    log_p_ucache_fast, log_p_ucache_slow, log_p_attn_fast, log_p_attn_slow, targets,
    weights, compute_topk=False
):
    """Linear prob-space blend: P = sum_k w_k * P_k (11-component version)."""
    weights = np.array(weights, dtype=np.float32)
    w_sum = weights.sum()
    if w_sum <= 0:
        return {"ppl": float("inf"), "top1": 0.0, "top5": 0.0, "n": int(len(targets))}
    if abs(w_sum - 1.0) > 1e-6:
        weights = weights / w_sum

    n = len(targets)
    idx = np.arange(n)
    
    components_t = [
        log_p_kn[idx, targets],
        log_p_mix[idx, targets],
        log_p_tri_fast[idx, targets],
        log_p_tri_slow[idx, targets],
        log_p_bi_fast[idx, targets],
        log_p_bi_slow[idx, targets],
        log_p_ucache_fast[idx, targets],
        log_p_ucache_slow[idx, targets],
        log_p_attn_fast[idx, targets],
        log_p_attn_slow[idx, targets]
    ]

    parts = []
    for w, lp_t in zip(weights, components_t):
        if w <= 0:
            continue
        parts.append(math.log(w) + lp_t)
        
    stack = np.stack(parts, axis=0)
    m = stack.max(axis=0)
    log_p_target = m + np.log(np.exp(stack - m[None]).sum(axis=0))
    nll = -log_p_target.sum()
    ppl = math.exp(nll / n)

    if not compute_topk:
        return {"ppl": ppl, "top1": None, "top5": None, "n": int(n)}

    c1 = 0
    c5 = 0
    BATCH = 2048
    active_components = [
        (w, lp) for w, lp in zip(
            weights, [
                log_p_kn, log_p_mix, log_p_tri_fast, log_p_tri_slow,
                log_p_bi_fast, log_p_bi_slow, log_p_ucache_fast, log_p_ucache_slow,
                log_p_attn_fast, log_p_attn_slow
            ]
        ) if w > 0
    ]
    for st in range(0, n, BATCH):
        e = min(st + BATCH, n)
        bparts = [math.log(w) + lp[st:e] for w, lp in active_components]
        sk = np.stack(bparts, axis=0)
        mm = sk.max(axis=0)
        lp_batch = mm + np.log(np.exp(sk - mm[None]).sum(axis=0))
        am = np.argmax(lp_batch, axis=1)
        tgt = targets[st:e]
        c1 += (am == tgt).sum()
        top5 = np.argpartition(-lp_batch, 5, axis=1)[:, :5]
        c5 += (top5 == tgt[:, None]).any(axis=1).sum()
        
    return {"ppl": ppl, "top1": float(c1) / n, "top5": float(c5) / n, "n": int(n)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kn-pickle", type=str, required=True)
    p.add_argument("--counts-file", type=str, required=True)
    p.add_argument("--K-pos", type=int, default=2)
    p.add_argument("--top-M", type=int, default=16)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--alpha-mix", type=float, default=0.01)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--alpha-tri-fast", type=float, default=1e-5)
    p.add_argument("--alpha-tri-slow", type=float, default=1e-5)
    p.add_argument("--alpha-bi-fast", type=float, default=1e-5)
    p.add_argument("--alpha-bi-slow", type=float, default=1e-5)
    p.add_argument("--alpha-ucache-fast", type=float, default=1e-5)
    p.add_argument("--alpha-ucache-slow", type=float, default=1e-5)
    p.add_argument("--lam-tri-fast", type=float, default=0.002)
    p.add_argument("--lam-tri-slow", type=float, default=0.0002)
    p.add_argument("--lam-bi-fast", type=float, default=0.005)
    p.add_argument("--lam-bi-slow", type=float, default=0.0005)
    p.add_argument("--lam-ucache-fast", type=float, default=0.002)
    p.add_argument("--lam-ucache-slow", type=float, default=0.0002)
    
    # continuous attention cache params
    p.add_argument("--W-attn-fast", type=int, default=1000)
    p.add_argument("--beta-attn-fast", type=float, default=15.0)
    p.add_argument("--theta-attn-fast", type=float, default=0.02)
    p.add_argument("--alpha-attn-fast", type=float, default=1e-5)
    
    p.add_argument("--W-attn-slow", type=int, default=3000)
    p.add_argument("--beta-attn-slow", type=float, default=8.0)
    p.add_argument("--theta-attn-slow", type=float, default=0.002)
    p.add_argument("--alpha-attn-slow", type=float, default=1e-5)
    
    p.add_argument("--train-tokens", type=str, default="22M")
    p.add_argument("--val-tokens", type=str, default="30K")
    p.add_argument("--eval-tokens", type=str, default="100K")
    p.add_argument("--tag", type=str, default="default")
    args = p.parse_args()

    train_n = parse_size(args.train_tokens)
    val_n = parse_size(args.val_tokens)
    eval_n = parse_size(args.eval_tokens)

    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    emb_dev = emb.to(DEVICE)
    ids = load_or_build_tokens(bpe, bpe_to_lm, V)
    ids_np = ids.numpy().astype(np.int32)
    T = len(ids)
    if train_n + val_n + eval_n > T:
        train_n = max(T - val_n - eval_n, T // 2)
    val_ids_t = ids[train_n:train_n + val_n]
    val_ids_n = ids_np[train_n:train_n + val_n]
    eval_ids_t = ids[-eval_n:]
    eval_ids_n = ids_np[-eval_n:]

    print(f"[v30] 11-Way Multi-Temporal Decayed Cache + Continuous Attention Retrieval Caches")
    print(f"[split] train={train_n:,}  val={val_n:,}  eval={eval_n:,}")

    print(f"[load] KN  {args.kn_pickle}")
    with open(args.kn_pickle, "rb") as f:
        kn = pickle.load(f)
        print(f"[load] counts  {args.counts_file}")
    blob = torch.load(args.counts_file, map_location=DEVICE, weights_only=False)
    mu = blob["mu"].to(DEVICE)
    counts = blob["counts"].to(DEVICE)
    assert blob["K_pos"] == args.K_pos and blob["V"] == V
    model = SparseMixtureClusterLM.from_counts(mu, counts, alpha=args.alpha_mix,
                                                V=V, K_pos=args.K_pos, d_emb=d)

    @torch.no_grad()
    def compute_log_p_mix_low_mem(ids_t: torch.Tensor, emb_dev_t: torch.Tensor, model_t: SparseMixtureClusterLM,
                                   K_pos_t: int, top_M_t: int, tau_t: float, gamma_t: float) -> torch.Tensor:
        V_t = model_t.V
        N_t = ids_t.shape[0]
        r = build_residual(ids_t.to(emb_dev_t.device).long(), emb_dev_t, K_pos_t)  # (N, d)
        out_chunks = []
        start_t = K_pos_t - 1
        end_t = N_t - 1
        mu_sq = (model_t.mu * model_t.mu).sum(dim=1)
        chunk_size = 400
        for s in range(start_t, end_t, chunk_size):
            e = min(s + chunk_size, end_t)
            r_c = r[s:e].to(DEVICE)
            r_sq = (r_c * r_c).sum(dim=1, keepdim=True)
            d2 = r_sq + mu_sq.unsqueeze(0) - 2 * (r_c @ model_t.mu.T)  # (B, K_cl)
            if top_M_t and top_M_t < model_t.mu.shape[0]:
                _, idx = d2.topk(top_M_t, dim=1, largest=False)
                d2_top = d2.gather(1, idx)
                log_pi = F.log_softmax(-d2_top / tau_t, dim=1)
                log_p_top = model_t.log_p_cluster[idx].float()
                log_mix = torch.logsumexp(log_pi.unsqueeze(2) + log_p_top, dim=1)
            else:
                log_pi = F.log_softmax(-d2 / tau_t, dim=1)
                log_mix = torch.logsumexp(log_pi.unsqueeze(2) + model_t.log_p_cluster.float().unsqueeze(0), dim=1)
            if gamma_t < 1.0:
                log_p = torch.logaddexp(
                    math.log(gamma_t) + log_mix,
                    math.log(1 - gamma_t) + model_t.log_p_uni.float().unsqueeze(0),
                )
            else:
                log_p = log_mix
            out_chunks.append(log_p.cpu())
            
        return torch.cat(out_chunks, dim=0)

    def prepare(ids_t, ids_n, label):
        print(f"\n[{label}] computing 10 underlying component log-prob tables")
        t0 = time.time()
        log_p_mix = compute_log_p_mix_low_mem(ids_t, emb_dev, model, args.K_pos,
                                               args.top_M, args.tau, args.gamma).numpy()
        print(f"  mix done ({time.time() - t0:.1f}s, shape={log_p_mix.shape})")
        
        t0 = time.time()
        log_p_kn = compute_log_p_kn(kn, ids_n, args.K_pos)
        print(f"  KN done ({time.time() - t0:.1f}s)")
        
        t0 = time.time()
        log_p_trif, log_p_tris, log_p_bif, log_p_bis, log_p_ucf, log_p_ucs = build_v30_induction_log_probs(
            ids_n, V, args.K_pos, args.window, 
            args.lam_tri_fast, args.lam_tri_slow,
            args.lam_bi_fast, args.lam_bi_slow, 
            args.lam_ucache_fast, args.lam_ucache_slow,
            args.alpha_tri_fast, args.alpha_tri_slow,
            args.alpha_bi_fast, args.alpha_bi_slow, 
            args.alpha_ucache_fast, args.alpha_ucache_slow
        )
        print(f"  standard decay caches done ({time.time() - t0:.1f}s)")
        
        # New Continuous Attention Retrieval Caches (Semantic Induction)
        t0 = time.time()
        print(f"  computing fast episodic attention cache...")
        log_p_attn_f = compute_log_p_attn(
            ids_t, emb_dev, args.W_attn_fast, args.beta_attn_fast, args.theta_attn_fast, args.alpha_attn_fast, args.K_pos
        ).numpy()
        
        print(f"  computing slow episodic attention cache...")
        log_p_attn_s = compute_log_p_attn(
            ids_t, emb_dev, args.W_attn_slow, args.beta_attn_slow, args.theta_attn_slow, args.alpha_attn_slow, args.K_pos
        ).numpy()
        print(f"  attention caches done ({time.time() - t0:.1f}s)")
        
        targets = ids_n[args.K_pos:]
        return (log_p_kn, log_p_mix, log_p_trif, log_p_tris, log_p_bif, log_p_bis, 
                log_p_ucf, log_p_ucs, log_p_attn_f, log_p_attn_s, targets)

    # VAL
    (log_p_kn_v, log_p_mix_v, log_p_trif_v, log_p_tris_v, log_p_bif_v, log_p_bis_v, 
     log_p_ucf_v, log_p_ucs_v, log_p_attn_fv, log_p_attn_sv, targets_v) = prepare(val_ids_t, val_ids_n, "val")

    print(f"\n[val] Dirichlet random simplex sweep of 20,000 points to optimize 10 blend weights")
    # Sample weights from Dirichlet distribution concentrated around a strong prior
    # Components: [kn, mix, trif, tris, bif, bis, ucf, ucs, att_f, att_s]
    alpha_dir = [6.0, 0.4, 0.2, 0.2, 0.2, 1.2, 0.3, 0.3, 0.8, 0.8]  # concentrate near good configuration
    num_samples = 20000
    
    # generate random simplex configurations
    np.random.seed(42)
    dirichlet_samples = np.random.dirichlet(alpha_dir, size=num_samples)
    
    # include baseline ones explicitly
    extra_configs = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # KN-only
        [0.69, 0.03, 0.03, 0.0, 0.03, 0.15, 0.04, 0.03, 0.0, 0.0],  # v28 representation
        [0.71, 0.03, 0.01, 0.01, 0.02, 0.14, 0.04, 0.04, 0.0, 0.0],  # v29 best
    ]
    grid = np.concatenate([extra_configs, dirichlet_samples], axis=0)
    
    val_results = []
    t0 = time.time()
    for idx_cfg, w in enumerate(grid):
        r = eval_blend11(
            log_p_kn_v, log_p_mix_v, log_p_trif_v, log_p_tris_v, log_p_bif_v, log_p_bis_v, 
            log_p_ucf_v, log_p_ucs_v, log_p_attn_fv, log_p_attn_sv, targets_v, w
        )
        val_results.append((r["ppl"], w))
        
        if (idx_cfg + 1) % 5000 == 0:
            print(f"    evaluated {idx_cfg+1} weight vectors on val set ({time.time() - t0:.1f}s)")
            
    val_results.sort(key=lambda x: x[0])
    print(f"  Directory sweep finished. Top-10 val configs:")
    for b_ppl, b_w in val_results[:10]:
        print(f"    PPL={b_ppl:7.2f} w=(kn{b_w[0]:.3f}, mix{b_w[1]:.3f}, tf{b_w[2]:.3f}, ts{b_w[3]:.3f}, bf{b_w[4]:.3f}, bs{b_w[5]:.3f}, uf{b_w[6]:.3f}, us{b_w[7]:.3f}, af{b_w[8]:.3f}, as{b_w[9]:.3f})")
        
    best_val_ppl, best_w = val_results[0]
    print(f"\n[val] best PPL={best_val_ppl:.2f}")

    # Free memory
    del (log_p_kn_v, log_p_mix_v, log_p_trif_v, log_p_tris_v, log_p_bif_v, log_p_bis_v, 
         log_p_ucf_v, log_p_ucs_v, log_p_attn_fv, log_p_attn_sv, targets_v)

    # HELDOUT EVALUATION
    (log_p_kn_e, log_p_mix_e, log_p_trif_e, log_p_tris_e, log_p_bif_e, log_p_bis_e, 
     log_p_ucf_e, log_p_ucs_e, log_p_attn_fe, log_p_attn_se, targets_e) = prepare(eval_ids_t, eval_ids_n, "eval")

    print(f"\n[eval] HELDOUT — evaluating best configurations")
    eval_results = {}
    
    # We'll evaluate:
    # 1. KN-only
    # 2. v28 exact
    # 3. v29 best representation
    # 4. Our best-val config (v30)
    # plus the top 3 alternative configurations
    configs_to_test = [
        ("KN-only", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("v28-best", [0.69, 0.03, 0.03, 0.0, 0.03, 0.15, 0.04, 0.03, 0.0, 0.0]),
        ("v29-best", [0.71, 0.03, 0.01, 0.01, 0.02, 0.14, 0.04, 0.04, 0.0, 0.0]),
        ("v30-best-val", best_w)
    ]
    
    # Add top 3 other configurations
    added = 0
    for ppl, w in val_results[1:]:
        if added >= 3:
            break
        # make sure it's not too similar to best_w
        if np.abs(np.array(w) - np.array(best_w)).max() > 0.02:
            configs_to_test.append((f"v30-alt-{added+1}", w))
            added += 1

    for label, w in configs_to_test:
        r = eval_blend11(
            log_p_kn_e, log_p_mix_e, log_p_trif_e, log_p_tris_e, log_p_bif_e, log_p_bis_e, 
            log_p_ucf_e, log_p_ucs_e, log_p_attn_fe, log_p_attn_se, targets_e, w, compute_topk=True
        )
        print(f"  {label:15s} PPL={r['ppl']:7.3f}  top1={r['top1']*100:.2f}%  top5={r['top5']*100:.2f}%")
        eval_results[label] = {**r, "w": list(map(float, w))}

    out = {
        "model": "v30 11-way blend with Dual-temporal Trigram, Bigram, Unigram decayed caches + Semantic Attention Retrieval Cache",
        "W_attn_fast": args.W_attn_fast, "beta_attn_fast": args.beta_attn_fast, "theta_attn_fast": args.theta_attn_fast,
        "W_attn_slow": args.W_attn_slow, "beta_attn_slow": args.beta_attn_slow, "theta_attn_slow": args.theta_attn_slow,
        "best_val_w": list(map(float, best_w)),
        "eval_heldout": eval_results,
        "val_top10": [(float(ppl), list(map(float, w))) for ppl, w in val_results[:10]],
    }
    out_path = ARTIFACT / f"eval_results_v30_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()