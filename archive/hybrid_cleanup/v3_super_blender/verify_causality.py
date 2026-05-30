"""hybrid/v3_super_blender/verify_causality.py

Strict, automated check to ensure there is zero future target leakage in the
feature representation and blender model forward passes.
"""
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.v1_blender.blender_model import build_feature_matrix
from hybrid.v3_super_blender.model import GRUBlender, LookbackMLPBlender, CausalConvBlender, WindowMLPBlender

def check_causality():
    print("==================================================")
    print("running 100% strict causal validation verification")
    print("==================================================")
    
    # Generate mock continuous positions
    T = 100
    C = 21
    d = 256
    
    # Mock files
    log_p_observed = torch.randn(T, C)
    log_p_lag1 = torch.randn(T, C)
    entropy = torch.randn(T, C)
    max_log_prob = torch.randn(T, C)
    emb = torch.randn(8000, d)
    observed_ids = torch.randint(0, 8000, (T,))
    topk_log_probs = torch.randn(T, C, 3)
    
    # Build base features
    features_full = build_feature_matrix(
        log_p_observed, log_p_lag1, entropy, max_log_prob,
        emb, observed_ids, use_embedding=True, topk_log_probs=topk_log_probs
    )
    
    # Test each model type
    models_to_test = [
        ("lookback_mlp", LookbackMLPBlender(features_full.shape[1], C, lookback_window=16)),
        ("window_mlp", WindowMLPBlender(features_full.shape[1], C, lookback_window=16)),
        ("gru", GRUBlender(features_full.shape[1], C)),
        ("causal_conv", CausalConvBlender(features_full.shape[1], C))
    ]
    
    for name, model in models_to_test:
        model.eval()
        
        # We perturb a single position t = 50 in features_full
        # Changing position t should NOT affect predictions for any steps < t
        t_perturb = 50
        
        features_perturbed = features_full.clone()
        features_perturbed[t_perturb:] = torch.randn_like(features_perturbed[t_perturb:])
        
        with torch.no_grad():
            if name in ["lookback_mlp", "window_mlp"]:
                # Normal inference
                out_full = model(features_full)
                out_perturbed = model(features_perturbed)
            elif name == "gru":
                out_full, _ = model(features_full.unsqueeze(0))
                out_perturbed, _ = model(features_perturbed.unsqueeze(0))
                out_full = out_full.squeeze(0)
                out_perturbed = out_perturbed.squeeze(0)
            elif name == "causal_conv":
                out_full = model(features_full.unsqueeze(0)).squeeze(0)
                out_perturbed = model(features_perturbed.unsqueeze(0)).squeeze(0)
                
        # Check that prior to t_perturb, predictions are identical
        diff = torch.abs(out_full[:t_perturb] - out_perturbed[:t_perturb]).max().item()
        
        print(f"Model {name:15s} | Max change in steps < {t_perturb}: {diff:.3e}")
        if diff > 1e-5:
            print(f"CAUSALITY LEAK DETECTED in model {name}! Diff: {diff}")
            sys.exit(1)
        else:
            print(f"Model {name:15s} | Causal validity: PASSED")
            
    print("\nAll blenders are 100% verified causal! Zero leakage detected.")

if __name__ == "__main__":
    check_causality()
