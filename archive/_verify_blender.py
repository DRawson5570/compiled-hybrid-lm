"""Verify compiled blender PPL and set up honest evaluation infrastructure."""
import sys, os
# Force deepseek_experiments hybrid to load first
for k in list(sys.modules):
    if 'hybrid' in k or 'compile_wiki' in k:
        del sys.modules[k]
sys.path.insert(0, '/home/drawson/deepseek_experiments')
sys.path.append('/home/drawson/llm_decoupling')

import numpy as np, torch, math
from hybrid.v3_super_blender.model import WindowMLPBlender
from hybrid.v1_blender.blender_model import build_feature_matrix
from compile_wiki_lm_v13 import load_setup, load_or_build_tokens

DEEPSEEK = '/home/drawson/deepseek_experiments/hybrid'

# Load embeddings and data
_bpe, _vocab, _tok2id, _bpe_to_lm, emb, V, d = load_setup()
emb = emb.float()
print(f'V={V}, d={d}')

# Load the WindowMLP blender (best compiled model)
val = np.load(f'{DEEPSEEK}/v3_super_blender/data_real_v33/val.npz', allow_pickle=True)
eval_data = np.load(f'{DEEPSEEK}/v3_super_blender/data_real_v33/eval.npz', allow_pickle=True)
print(f'val: {val["log_p_targets"].shape[0]} tokens, eval: {eval_data["log_p_targets"].shape[0]} tokens')

# Build features
val_feat = build_feature_matrix(
    torch.tensor(val['log_p_observed']), torch.tensor(val['log_p_lag1']),
    torch.tensor(val['entropy']), torch.tensor(val['max_log_prob']),
    emb, torch.tensor(val['observed']), use_embedding=True
).float()

eval_feat = build_feature_matrix(
    torch.tensor(eval_data['log_p_observed']), torch.tensor(eval_data['log_p_lag1']),
    torch.tensor(eval_data['entropy']), torch.tensor(eval_data['max_log_prob']),
    emb, torch.tensor(eval_data['observed']), use_embedding=True
).float()

print(f'Feature dim: {val_feat.shape[1]}')

# Load blender
ckpt = torch.load(f'{DEEPSEEK}/v3_super_blender/saved_models_v33/blender_window_mlp.pt', map_location='cpu')
blender = WindowMLPBlender(
    single_step_dim=val_feat.shape[1], n_channels=21,
    lookback_window=16, hidden=256, dropout=0.1, init_uniform=False
)
blender.load_state_dict(ckpt['state_dict'])
blender.eval()

# Compute compiled blend per-target log-probs
with torch.no_grad():
    log_w_val = blender(blender.build_windowed_features(val_feat), is_already_windowed=True)
    log_w_eval = blender(blender.build_windowed_features(eval_feat), is_already_windowed=True)

log_p_targets_val = torch.tensor(val['log_p_targets'])
log_p_targets_eval = torch.tensor(eval_data['log_p_targets'])

compiled_val = torch.logsumexp(log_w_val + log_p_targets_val, dim=-1)
compiled_eval = torch.logsumexp(log_w_eval + log_p_targets_eval, dim=-1)

print(f'\n=== Compiled WindowMLP Blend ===')
print(f'Val PPL: {math.exp(-compiled_val.mean().item()):.4f}')
print(f'Val NLL: {-compiled_val.mean().item():.4f}')
print(f'Eval PPL: {math.exp(-compiled_eval.mean().item()):.4f}')
print(f'Eval NLL: {-compiled_eval.mean().item():.4f}')

# Per-channel baselines
print(f'\n=== Per-channel PPL (eval) ===')
for c in range(21):
    ch_nll = -log_p_targets_eval[:, c].mean().item()
    print(f'  ch{c:2d} {eval_data["channel_names"][c]:8s}  PPL={math.exp(ch_nll):8.2f}')

# Best single channel
best_ch = log_p_targets_eval.max(dim=1).values.mean().item()
print(f'\n  Oracle (best per-token): PPL={math.exp(-best_ch):.4f}')
# Uniform mix
uni = torch.logsumexp(torch.full_like(log_p_targets_eval, -math.log(21)), dim=-1)
uni_blend = torch.logsumexp(uni.unsqueeze(-1) + log_p_targets_eval, dim=-1)
print(f'  Uniform mix: PPL={math.exp(-uni_blend.mean().item()):.4f}')
