"""
compile_wiki_lm_v32.py — 18-way multi-temporal decayed conditional + unconditioned induction
                          + Multi-Scale Unigram & Context Residual Attention Caches Ensemble (Pushing PPL < 29.0)
=============================================================================================================

This file constructs an enterprise-grade 18-way model blend combining:
- Global Kneser-Ney 7-Gram
- Semi-Parametric Positional Mixture Clusters
- Dual-temporal Trigram Decayed Caches (Fast/Slow)
- Dual-temporal Bigram Decayed Caches (Fast/Slow)
- Dual-temporal Unconditioned Unigram Decayed Caches (Fast/Slow)
- 10 semantic attention retrieval caches:
  - 3 Unigram Space Attention Caches (Fast, Slow, Global)
  - 2 Residual Space (K=1, Bigram) Caches (Fast, Slow)
  - 3 Residual Space (K=2, Trigram) Caches (Fast, Slow, Global)
  - 2 Residual Space (K=3, Fourgram) Caches (Fast, Slow)

Optimized locally using high-performance chunk-based sliding-window matrix-vector logic on CUDA.
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

ARTIFACT = Path("/home/drawson/deepseek_experiments/artifacts/compiled_wiki_lm_v32")
ARTIFACT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def build_v32_induction_log_probs(
    ids_np: np.ndarray, V: int, K_pos: int, window: int,
    lam_tri_fast: float, lam_tri_slow: float,
    lam_bi_fast: float, lam_bi_slow: float,
    lam_ucache_fast: float, lam_ucache_slow: float,
    alpha_tri_fast: float, alpha_tri_slow: float,
    alpha_bi_fast: float, alpha_bi_slow: float,
    alpha_ucache_fast: float, alpha_ucache_slow: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream ids; build sliding-window dual decayed trigram, dual conditional bigram and dual unconditioned unigram cache."""
    N = len(ids_np)
    T = N - K_pos

    # 1. Decayed Bigram - Fast
    B_bi_fast = torch.zeros(V, V, device=DEVICE, dtype=torch.float32)
    R_bi_fast = torch.zeros(V, device=DEVICE, dtype=torch.float32)
    last_bi_fast = torch.zeros(V, V, device=DEVICE, dtype=torch.int32)
    last_R_bi_fast = torch.zeros(V, device=DEVICE, dtype=torch.int32)

    # 2. Decayed Bigram - Slow
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

        # F. Update
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

        if t >= 2:
            a_val = int(ids_np[t - 2])
            key_val = (a_val, b)
            
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

        history.append((a_val if t >= 2 else None, b, c, key_val if t >= 2 else None))

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

        if (i + 1) % 15000 == 0:
            print(f"    [ind] {i+1}/{T} ({time.time() - t0:.1f}s)")

    return log_p_tri_fast, log_p_tri_slow, log_p_bi_fast, log_p_bi_slow, log_p_ucache_fast, log_p_ucache_slow


@torch.no_grad()
def compute_log_p_attn_unigram(
    ids_t: torch.Tensor, emb_dev: torch.Tensor, W_attn: int, beta: float, theta: float, alpha_attn: float, K_pos: int
) -> torch.Tensor:
    """Compute semantic Continuous Attention Cache (Unigram Word level)."""
    N = ids_t.shape[0]
    out_log_p = torch.empty((N - K_pos, emb_dev.shape[0]), dtype=torch.float32)
    emb_norm = F.normalize(emb_dev, p=2, dim=1)
    
    chunk_size = 2000
    for s in range(K_pos, N, chunk_size):
        e = min(s + chunk_size, N)
        B = e - s
        
        q = emb_norm[ids_t[s:e].long()].to(DEVICE)
        p_attn_batch = torch.zeros((B, emb_dev.shape[0]), device=DEVICE, dtype=torch.float32)
        
        for idx in range(B):
            t = s + idx
            j_start = max(0, t - W_attn)
            j_end = t
            W_current = j_end - j_start
            if W_current == 0:
                continue
                
            context_ids = ids_t[j_start:j_end].long().to(DEVICE)
            next_ids = ids_t[j_start+1:j_end+1].long().to(DEVICE)
            
            k = emb_norm[context_ids]
            dot = torch.mv(k, q[idx])
            
            if theta > 0.0:
                dist = torch.arange(W_current, 0, -1, device=DEVICE, dtype=torch.float32)
                scores = torch.exp(beta * dot - theta * dist)
            else:
                scores = torch.exp(beta * dot)
                
            scores_sum = scores.sum()
            if scores_sum > 0:
                probs = scores / scores_sum
                p_attn_batch[idx].scatter_add_(0, next_ids, probs)
                
        p_smoothed = (p_attn_batch + alpha_attn) / (1.0 + alpha_attn * emb_dev.shape[0])
        out_log_p[s-K_pos : e-K_pos] = torch.log(p_smoothed.clamp_min(1e-30)).cpu()
        
    return out_log_p


@torch.no_grad()
def compute_log_p_attn_residual_sliced(
    ids_t: torch.Tensor, r_full: torch.Tensor, d: int, K: int, W_attn: int, beta: float, theta: float, alpha_attn: float, K_pos: int, V: int
) -> torch.Tensor:
    """Compute semantic State Attention Cache (Context Phrase level) using precomputed core residual slices."""
    N = ids_t.shape[0]
    
    # Slice the multi-token representation for K_pos matching
    dim_res = (K + 1) * d
    r = r_full[:, :dim_res]
    r_norm = F.normalize(r, p=2, dim=1)
    
    out_log_p = torch.empty((N - K_pos, V), dtype=torch.float32)
    chunk_size = 2000
    for s in range(K_pos, N, chunk_size):
        e = min(s + chunk_size, N)
        B = e - s
        
        q = r_norm[s:e].to(DEVICE)
        p_attn_batch = torch.zeros((B, V), device=DEVICE, dtype=torch.float32)
        
        for idx in range(B):
            t = s + idx
            j_start = max(0, t - W_attn)
            j_end = t
            W_current = j_end - j_start
            if W_current == 0:
                continue
                
            k = r_norm[j_start:j_end].to(DEVICE)
            dot = torch.mv(k, q[idx])
            
            next_ids = ids_t[j_start+1:j_end+1].long().to(DEVICE)
            if theta > 0.0:
                dist = torch.arange(W_current, 0, -1, device=DEVICE, dtype=torch.float32)
                scores = torch.exp(beta * dot - theta * dist)
            else:
                scores = torch.exp(beta * dot)
                
            scores_sum = scores.sum()
            if scores_sum > 0:
                probs = scores / scores_sum
                p_attn_batch[idx].scatter_add_(0, next_ids, probs)
                
        p_smoothed = (p_attn_batch + alpha_attn) / (1.0 + alpha_attn * V)
        out_log_p[s-K_pos : e-K_pos] = torch.log(p_smoothed.clamp_min(1e-30)).cpu()
        
    return out_log_p


def eval_blend_C(log_p_list, targets, weights, compute_topk=False):
    """Linear prob-space blend: P = sum_k w_k * P_k for C components."""
    weights = np.array(weights, dtype=np.float32)
    w_sum = weights.sum()
    if w_sum <= 0:
        return {"ppl": float("inf"), "top1": 0.0, "top5": 0.0, "n": int(len(targets))}
    if abs(w_sum - 1.0) > 1e-6:
        weights = weights / w_sum

    n = len(targets)
    idx = np.arange(n)
    
    # Extract only active components to save massive exponentiation steps
    active_idx = [k for k, w in enumerate(weights) if w > 1e-5]
    if len(active_idx) == 0:
        return {"ppl": float("inf"), "top1": 0.0, "top5": 0.0, "n": int(n)}
        
    parts = []
    for k in active_idx:
        w = weights[k]
        lp_t = log_p_list[k][idx, targets]
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
    active_lps = [(weights[k], log_p_list[k]) for k in active_idx]
    for st in range(0, n, BATCH):
        e = min(st + BATCH, n)
        bparts = [math.log(w) + lp[st:e] for w, lp in active_lps]
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
    
    # Unigram Attention Caches
    p.add_argument("--W-attn-uf", type=int, default=2000)
    p.add_argument("--beta-attn-uf", type=float, default=14.0)
    p.add_argument("--theta-attn-uf", type=float, default=0.02)
    p.add_argument("--alpha-attn-uf", type=float, default=1e-5)
    
    p.add_argument("--W-attn-us", type=int, default=8000)
    p.add_argument("--beta-attn-us", type=float, default=8.0)
    p.add_argument("--theta-attn-us", type=float, default=0.002)
    p.add_argument("--alpha-attn-us", type=float, default=1e-5)

    p.add_argument("--W-attn-ug", type=int, default=16384)
    p.add_argument("--beta-attn-ug", type=float, default=10.0)
    p.add_argument("--theta-attn-ug", type=float, default=0.0)  # Pure Global
    p.add_argument("--alpha-attn-ug", type=float, default=1e-5)
    
    # Residual Attention Caches (K=1, Bigram space)
    p.add_argument("--W-attn-rf1", type=int, default=2000)
    p.add_argument("--beta-attn-rf1", type=float, default=16.0)
    p.add_argument("--theta-attn-rf1", type=float, default=0.02)
    p.add_argument("--alpha-attn-rf1", type=float, default=1e-5)
    
    p.add_argument("--W-attn-rs1", type=int, default=8000)
    p.add_argument("--beta-attn-rs1", type=float, default=10.0)
    p.add_argument("--theta-attn-rs1", type=float, default=0.002)
    p.add_argument("--alpha-attn-rs1", type=float, default=1e-5)

    # Residual Attention Caches (K=2, Trigram space)
    p.add_argument("--W-attn-rf2", type=int, default=2000)
    p.add_argument("--beta-attn-rf2", type=float, default=18.0)
    p.add_argument("--theta-attn-rf2", type=float, default=0.03)
    p.add_argument("--alpha-attn-rf2", type=float, default=1e-5)
    
    p.add_argument("--W-attn-rs2", type=int, default=8000)
    p.add_argument("--beta-attn-rs2", type=float, default=12.0)
    p.add_argument("--theta-attn-rs2", type=float, default=0.003)
    p.add_argument("--alpha-attn-rs2", type=float, default=1e-5)

    p.add_argument("--W-attn-rg2", type=int, default=16384)
    p.add_argument("--beta-attn-rg2", type=float, default=14.0)
    p.add_argument("--theta-attn-rg2", type=float, default=0.0)  # Pure Global
    p.add_argument("--alpha-attn-rg2", type=float, default=1e-5)

    # Residual Attention Caches (K=3, Fourgram space)
    p.add_argument("--W-attn-rf3", type=int, default=2000)
    p.add_argument("--beta-attn-rf3", type=float, default=20.0)
    p.add_argument("--theta-attn-rf3", type=float, default=0.04)
    p.add_argument("--alpha-attn-rf3", type=float, default=1e-5)
    
    p.add_argument("--W-attn-rs3", type=int, default=8000)
    p.add_argument("--beta-attn-rs3", type=float, default=14.0)
    p.add_argument("--theta-attn-rs3", type=float, default=0.004)
    p.add_argument("--alpha-attn-rs3", type=float, default=1e-5)
    
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

    print(f"[v32] ENTERPRISE 18-Way Mixture Model: Decayed dynamic count + Multi-scale semantic attention retrieval")
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
        r = build_residual(ids_t.to(emb_dev_t.device).long(), emb_dev_t, K_pos_t)
        out_chunks = []
        start_t = K_pos_t - 1
        end_t = N_t - 1
        mu_sq = (model_t.mu * model_t.mu).sum(dim=1)
        chunk_size = 400
        for s in range(start_t, end_t, chunk_size):
            e = min(s + chunk_size, end_t)
            r_c = r[s:e].to(DEVICE)
            r_sq = (r_c * r_c).sum(dim=1, keepdim=True)
            d2 = r_sq + mu_sq.unsqueeze(0) - 2 * (r_c @ model_t.mu.T)
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
        print(f"\n[{label}] computing 18 underlying component log-prob tables")
        
        # 1. Cluster mixture
        t0 = time.time()
        log_p_mix = compute_log_p_mix_low_mem(ids_t, emb_dev, model, args.K_pos,
                                               args.top_M, args.tau, args.gamma).numpy()
        print(f"  (1/18) mix done ({time.time() - t0:.1f}s, shape={log_p_mix.shape})")
        
        # 2. Global KN7
        t0 = time.time()
        log_p_kn = compute_log_p_kn(kn, ids_n, args.K_pos)
        print(f"  (2/18) KN done ({time.time() - t0:.1f}s)")
        
        # 3. Dynamic caches
        t0 = time.time()
        log_p_trif, log_p_tris, log_p_bif, log_p_bis, log_p_ucf, log_p_ucs = build_v32_induction_log_probs(
            ids_n, V, args.K_pos, args.window, 
            args.lam_tri_fast, args.lam_tri_slow,
            args.lam_bi_fast, args.lam_bi_slow, 
            args.lam_ucache_fast, args.lam_ucache_slow,
            args.alpha_tri_fast, args.alpha_tri_slow,
            args.alpha_bi_fast, args.alpha_bi_slow, 
            args.alpha_ucache_fast, args.alpha_ucache_slow
        )
        print(f"  (3-8/18) dynamic decay caches done ({time.time() - t0:.1f}s)")
        
        # 4. Unigram Attention Caches (uf, us, ug)
        t0 = time.time()
        print(f"  computing unigram attention caches...")
        log_p_attn_uf = compute_log_p_attn_unigram(
            ids_t, emb_dev, args.W_attn_uf, args.beta_attn_uf, args.theta_attn_uf, args.alpha_attn_uf, args.K_pos
        ).numpy()
        log_p_attn_us = compute_log_p_attn_unigram(
            ids_t, emb_dev, args.W_attn_us, args.beta_attn_us, args.theta_attn_us, args.alpha_attn_us, args.K_pos
        ).numpy()
        log_p_attn_ug = compute_log_p_attn_unigram(
            ids_t, emb_dev, args.W_attn_ug, args.beta_attn_ug, args.theta_attn_ug, args.alpha_attn_ug, args.K_pos
        ).numpy()
        print(f"  (9-11/18) unigram attention caches done ({time.time() - t0:.1f}s)")
        
        # 5. Core Phrase Residual construction (K=3 max)
        t0 = time.time()
        print(f"  computing core phrase residuals...")
        r_full = build_residual(ids_t.to(DEVICE).long(), emb_dev, K=3)
        
        # 6. Residual Attention Caches
        print(f"  computing multi-scale state attention caches...")
        
        # Residual K_pos=1 (Bigram)
        log_p_attn_rf1 = compute_log_p_attn_residual_sliced(
            ids_t, r_full, d, K=1, W_attn=args.W_attn_rf1, beta=args.beta_attn_rf1, theta=args.theta_attn_rf1, alpha_attn=args.alpha_attn_rf1, K_pos=args.K_pos, V=V
        ).numpy()
        log_p_attn_rs1 = compute_log_p_attn_residual_sliced(
            ids_t, r_full, d, K=1, W_attn=args.W_attn_rs1, beta=args.beta_attn_rs1, theta=args.theta_attn_rs1, alpha_attn=args.alpha_attn_rs1, K_pos=args.K_pos, V=V
        ).numpy()
        
        # Residual K_pos=2 (Trigram)
        log_p_attn_rf2 = compute_log_p_attn_residual_sliced(
            ids_t, r_full, d, K=2, W_attn=args.W_attn_rf2, beta=args.beta_attn_rf2, theta=args.theta_attn_rf2, alpha_attn=args.alpha_attn_rf2, K_pos=args.K_pos, V=V
        ).numpy()
        log_p_attn_rs2 = compute_log_p_attn_residual_sliced(
            ids_t, r_full, d, K=2, W_attn=args.W_attn_rs2, beta=args.beta_attn_rs2, theta=args.theta_attn_rs2, alpha_attn=args.alpha_attn_rs2, K_pos=args.K_pos, V=V
        ).numpy()
        log_p_attn_rg2 = compute_log_p_attn_residual_sliced(
            ids_t, r_full, d, K=2, W_attn=args.W_attn_rg2, beta=args.beta_attn_rg2, theta=args.theta_attn_rg2, alpha_attn=args.alpha_attn_rg2, K_pos=args.K_pos, V=V
        ).numpy()
        
        # Residual K_pos=3 (Fourgram)
        log_p_attn_rf3 = compute_log_p_attn_residual_sliced(
            ids_t, r_full, d, K=3, W_attn=args.W_attn_rf3, beta=args.beta_attn_rf3, theta=args.theta_attn_rf3, alpha_attn=args.alpha_attn_rf3, K_pos=args.K_pos, V=V
        ).numpy()
        log_p_attn_rs3 = compute_log_p_attn_residual_sliced(
            ids_t, r_full, d, K=3, W_attn=args.W_attn_rs3, beta=args.beta_attn_rs3, theta=args.theta_attn_rs3, alpha_attn=args.alpha_attn_rs3, K_pos=args.K_pos, V=V
        ).numpy()
        
        print(f"  (12-18/18) multi-scale state attention caches done ({time.time() - t0:.1f}s)")
        
        targets = ids_n[args.K_pos:]
        
        log_p_list = [
            log_p_kn, log_p_mix,
            log_p_trif, log_p_tris, log_p_bif, log_p_bis, log_p_ucf, log_p_ucs,
            log_p_attn_uf, log_p_attn_us, log_p_attn_ug,
            log_p_attn_rf1, log_p_attn_rs1,
            log_p_attn_rf2, log_p_attn_rs2, log_p_attn_rg2,
            log_p_attn_rf3, log_p_attn_rs3
        ]
        return log_p_list, targets

    # VAL
    log_p_list_v, targets_v = prepare(val_ids_t, val_ids_n, "val")

    print(f"\n[val] Dirichlet random simplex sweep of 60,000 points to optimize 18 blend weights")
    # Components: 
    # 0: kn, 1: mix
    # 2: tri_f, 3: tri_s
    # 4: bi_f, 5: bi_s
    # 6: uc_f, 7: uc_s
    # 8: att_uf, 9: att_us, 10: att_ug
    # 11: att_rf1, 12: att_rs1
    # 13: att_rf2, 14: att_rs2, 15: att_rg2
    # 16: att_rf3, 17: att_rs3
    alpha_dir = [
        6.0,  0.1,
        0.1,  0.1,
        0.1,  1.0,
        0.2,  0.2,
        0.6,  0.6,  0.4,
        0.8,  0.8,
        1.0,  1.0,  0.6,
        1.0,  1.0
    ]
    num_samples = 60000
    
    np.random.seed(42)
    dirichlet_samples = np.random.dirichlet(alpha_dir, size=num_samples)
    
    # Prepend some strong default configs from v28, v29, v31
    extra_configs = [
        [1.0] + [0.0]*17,
        [0.69, 0.03, 0.03, 0.0, 0.03, 0.15, 0.04, 0.03] + [0.0]*10,
        [0.71, 0.03, 0.01, 0.01, 0.02, 0.14, 0.04, 0.04] + [0.0]*10,
    ]
    grid = np.concatenate([extra_configs, dirichlet_samples], axis=0)
    
    val_results = []
    t0 = time.time()
    for idx_cfg, w in enumerate(grid):
        r = eval_blend_C(log_p_list_v, targets_v, w)
        val_results.append((r["ppl"], w))
        
        if (idx_cfg + 1) % 15000 == 0:
            print(f"    evaluated {idx_cfg+1} weight vectors on val set ({time.time() - t0:.1f}s)")
            
    val_results.sort(key=lambda x: x[0])
    print(f"  Directory sweep finished. Top-10 val configs:")
    labels_short = [
        "kn", "mix", "tf", "ts", "bf", "bs", "uf", "us", "a_uf", "a_us", "a_ug", "a_rf1", "a_rs1", "a_rf2", "a_rs2", "a_rg2", "a_rf3", "a_rs3"
    ]
    for b_ppl, b_w in val_results[:10]:
        w_pairs = [f"{lbl}:{val:.3f}" for lbl, val in zip(labels_short, b_w) if val > 0.005]
        print(f"    PPL={b_ppl:7.2f} w=(" + ", ".join(w_pairs) + ")")
        
    best_val_ppl, best_w = val_results[0]
    print(f"\n[val] best PPL={best_val_ppl:.2f}")

    # Free memory
    del log_p_list_v, targets_v

    # HELDOUT EVALUATION
    log_p_list_e, targets_e = prepare(eval_ids_t, eval_ids_n, "eval")

    print(f"\n[eval] HELDOUT — evaluating best configurations")
    eval_results = {}
    
    # Build v28 and v29 configs
    v28_w = [0.69, 0.03, 0.03, 0.0, 0.03, 0.15, 0.04, 0.03] + [0.0]*10
    v29_w = [0.71, 0.03, 0.01, 0.01, 0.02, 0.14, 0.04, 0.04] + [0.0]*10
    
    configs_to_test = [
        ("KN-only", [1.0] + [0.0]*17),
        ("v28-best", v28_w),
        ("v29-best", v29_w),
        ("v32-best-val", best_w)
    ]
    
    added = 0
    for ppl, w in val_results[1:]:
        if added >= 3:
            break
        if np.abs(np.array(w) - np.array(best_w)).max() > 0.02:
            configs_to_test.append((f"v32-alt-{added+1}", w))
            added += 1

    for label, w in configs_to_test:
        r = eval_blend_C(log_p_list_e, targets_e, w, compute_topk=True)
        print(f"  {label:15s} PPL={r['ppl']:7.3f}  top1={r['top1']*100:.2f}%  top5={r['top5']*100:.2f}%")
        eval_results[label] = {**r, "w": list(map(float, w))}

    out = {
        "model": "v32 18-way blend with dynamic decayed counts + multi-scale semantic unigram & context-residual attention caches",
        "W_uf": args.W_attn_uf, "beta_uf": args.beta_attn_uf, "theta_uf": args.theta_attn_uf,
        "W_us": args.W_attn_us, "beta_us": args.beta_attn_us, "theta_us": args.theta_attn_us,
        "W_ug": args.W_attn_ug, "beta_ug": args.beta_attn_ug, "theta_ug": args.theta_attn_ug,
        "W_rf1": args.W_attn_rf1, "beta_rf1": args.beta_attn_rf1, "theta_rf1": args.theta_attn_rf1,
        "W_rs1": args.W_attn_rs1, "beta_rs1": args.beta_attn_rs1, "theta_rs1": args.theta_attn_rs1,
        "W_rf2": args.W_attn_rf2, "beta_rf2": args.beta_attn_rf2, "theta_rf2": args.theta_attn_rf2,
        "W_rs2": args.W_attn_rs2, "beta_rs2": args.beta_attn_rs2, "theta_rs2": args.theta_attn_rs2,
        "W_rg2": args.W_attn_rg2, "beta_rg2": args.beta_attn_rg2, "theta_rg2": args.theta_attn_rg2,
        "W_rf3": args.W_attn_rf3, "beta_rf3": args.beta_attn_rf3, "theta_rf3": args.theta_attn_rf3,
        "W_rs3": args.W_attn_rs3, "beta_rs3": args.beta_attn_rs3, "theta_rs3": args.theta_attn_rs3,
        "best_val_w": list(map(float, best_w)),
        "eval_heldout": eval_results,
        "val_top10": [(float(ppl), list(map(float, w))) for ppl, w in val_results[:10]],
    }
    out_path = ARTIFACT / f"eval_results_v32_{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    main()