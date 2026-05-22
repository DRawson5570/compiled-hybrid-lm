"""CMI Blender Training — with channel signature features.

Key fix: add per-channel signature scores (log-prob of domain-specific tokens)
to the feature vector. This makes every channel identifiable by the blender.
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hybrid.v2_capabilities.channels import InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
from hybrid.v2_capabilities.retrieval_channel import RetrievalChannel, DEFAULT_DOCS
from hybrid.v2_capabilities.dataset import tok2id, id2tok, V, emb_dim, get_ppmi_embeddings
from hybrid.v1_blender.blender_model import mixture_nll
from hybrid.v3_super_blender.model import CausalConvBlender

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
torch.manual_seed(42)
random.seed(42)

emb = get_ppmi_embeddings().to(DEVICE)

# ─── Channel signature token IDs ───
SIG_TOKENS = {
    'instruct': ['chien', 'chat', 'pomme', 'gravité', 'force', 'mass', 'earth', 'attraction'],
    'reasoner': ['E0001', 'E0002', 'E0003', 'Therefore', 'larger', 'than'],
    'coder': ['def', 'return', 'import', 'numpy', 'as', 'np', 'zeros', '+', ':', '('],
    'tool': ['[USE_TOOL:', 'calculator', 'expr=', '54+23', '12+15', '77', '27', '300'],
    'retrieval': ['dog', 'cat', 'gravity', 'earth', 'french', 'apple'],
}
sig_ids = {k: torch.tensor([tok2id[t] for t in v if tok2id.get(t) is not None],
                           dtype=torch.long, device=DEVICE)
           for k, v in SIG_TOKENS.items()}
ch_names = ['instruct', 'reasoner', 'coder', 'tool', 'retrieval']

# ─── Build feature matrix with signatures ───
def build_features_with_sigs(p_outputs, ids, t: int):
    """Build feature vector with channel signature scores."""
    x_o = ids[t]
    x_l1 = ids[t - 1] if t > 0 else torch.zeros_like(ids[t])

    logp_o = torch.stack([p[t, x_o] for p in p_outputs])  # (C,)
    logp_l1 = torch.stack([p[t, x_l1] for p in p_outputs])
    ent = torch.stack([-(p[t].exp() * p[t]).sum() for p in p_outputs])
    mlp = torch.stack([p[t].max() for p in p_outputs])

    # Standard features: (4*C,)
    base = torch.cat([logp_o, logp_l1, ent, mlp])  # (4*C,)

    # Channel signature scores: avg log-prob of signature tokens per channel
    sigs = torch.zeros(len(ch_names), device=DEVICE)
    for ci, sn in enumerate(ch_names):
        tids = sig_ids[sn]
        if len(tids) > 0:
            sigs[ci] = p_outputs[ci][t, tids].mean()

    # Embedding of observed token
    emb_feat = emb[x_o]  # (d,)

    return torch.cat([base, sigs, emb_feat])  # (4*C + C + d,) = (4*5 + 5 + 16,) = 41


# ─── Generate training data ───
print("Building channels...", flush=True)
channels = [
    InstructChannel(tok2id, id2tok, emb),
    ReasonerChannel(tok2id, id2tok),
    CoderChannel(tok2id, id2tok),
    ToolChannel(tok2id, id2tok),
    RetrievalChannel(tok2id, id2tok, emb, doc_texts=DEFAULT_DOCS),
]
C = len(channels)
d = emb.shape[1]
F = 4 * C + C + d  # 20 + 5 + 16 = 41
print(f"C={C}, F={F}", flush=True)

# Rich prompts per channel
prompt_sets = {
    0: [  # Instruct
        ["translate", "dog", "to", "french"],
        ["translate", "cat", "to", "french"],
        ["translate", "apple", "to", "french"],
        ["translate", "gravity", "to", "french"],
        ["explain", "gravity"],
        ["explain", "dog"],
        ["explain", "cat"],
    ] * 50,
    1: [  # Reasoner
        ["E0001", "is", "larger", "than", "E0002", ".", "E0002", "is", "larger", "than", "E0003", ".", "Therefore", ",", "E0001", "is", "larger", "than"],
        ["E0004", "is", "larger", "than", "E0005", ".", "E0005", "is", "larger", "than", "E0006", ".", "Therefore", ",", "E0004", "is", "larger", "than"],
        ["E0007", "is", "larger", "than", "E0008", ".", "E0008", "is", "larger", "than", "E0009", ".", "Therefore", ",", "E0007", "is", "larger", "than"],
    ] * 100,
    2: [  # Coder
        ["def", "get_sum", "(", "a", ",", "b", ")", ":", "return", "a", "+", "b"],
        ["import", "numpy", "as", "np"],
        ["np", ".", "zeros", "(", "10", ")"],
    ] * 100,
    3: [  # Tool
        ["What", "is", "54", "+", "23", "?", "[USE_TOOL:", "calculator", "expr=", "54+23", "]", "Answer", "is", "77"],
        ["What", "is", "12", "+", "15", "?", "[USE_TOOL:", "calculator", "expr=", "12+15", "]", "Answer", "is", "27"],
        ["What", "is", "100", "+", "200", "?", "[USE_TOOL:", "calculator", "expr=", "100+200", "]", "Answer", "is", "300"],
    ] * 100,
    4: [  # Retrieval
        ["What", "is", "a", "dog"],
        ["What", "is", "a", "cat"],
        ["What", "is", "gravity"],
    ] * 100,
}

all_prompts = []
all_labels = []
for ch_idx, prompts in prompt_sets.items():
    for p in prompts:
        all_prompts.append(p)
        all_labels.append(ch_idx)

print(f"{len(all_prompts)} training sequences", flush=True)

# Build features
print("Building features...", flush=True)
feat_list, logp_list = [], []
for prompt, label in zip(all_prompts, all_labels):
    ids = torch.tensor([tok2id.get(t, tok2id["<UNK>"]) for t in prompt], device=DEVICE)
    T = len(prompt)
    p_outputs = [c.forward(ids).to(DEVICE) for c in channels]
    for t in range(T):
        feat = build_features_with_sigs(p_outputs, ids, t)
        feat_list.append(feat)
        tid = p_outputs[label][t].argmax().item()
        logp_list.append(torch.stack([p[t, tid] for p in p_outputs]))

features = torch.stack(feat_list).to(DEVICE)
log_p_targets = torch.stack(logp_list).to(DEVICE)
N = features.shape[0]
print(f"{N} training tokens, F={F}", flush=True)

# ─── Train ───
print("Training...", flush=True)
blender = CausalConvBlender(in_dim=F, n_channels=C, channels=64, kernel_size=3).to(DEVICE)
for p in blender.parameters():
    if p.dim() >= 2:
        nn.init.xavier_uniform_(p)

opt = torch.optim.AdamW(blender.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)

n_train = int(N * 0.8)
perm = torch.randperm(N)
f_tr, f_va = features[perm[:n_train]], features[perm[n_train:]]
l_tr, l_va = log_p_targets[perm[:n_train]], log_p_targets[perm[n_train:]]

best_val, best_ep = float("inf"), 0
best_state = None
patience = 300
no_improve = 0

for ep in range(2000):
    blender.train()
    for s in range(0, n_train, 64):
        j = torch.arange(s, min(s + 64, n_train), device=DEVICE)
        log_w = blender(f_tr[j].unsqueeze(0)).squeeze(0)
        nll = mixture_nll(log_w, l_tr[j]).mean()
        w_mean = log_w.exp().mean(dim=0)
        ent_bonus = -(w_mean * torch.log(w_mean + 1e-8)).sum() * 0.005
        loss = nll - ent_bonus
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(blender.parameters(), 1.0)
        opt.step()
    sched.step()

    blender.eval()
    with torch.no_grad():
        log_w = blender(f_va.unsqueeze(0)).squeeze(0)
        val_nll = mixture_nll(log_w, l_va).mean().item()
        avg_w = log_w[:min(50, len(log_w))].exp().mean(dim=0).tolist()

    if val_nll < best_val:
        best_val = val_nll
        best_ep = ep
        best_state = {k: v.clone() for k, v in blender.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if ep % 100 == 0:
        w_str = " ".join(f"{w:.2f}" for w in avg_w)
        print(f"  ep {ep:4d}: val={val_nll:.4f} w=[{w_str}] best={best_val:.4f}@{best_ep}", flush=True)

    if no_improve > patience:
        print(f"  Early stop at ep {ep}", flush=True)
        break

# Save
blender.load_state_dict(best_state)
save_path = ROOT / "artifacts" / "blender_5ch_sigs.pt"
save_path.parent.mkdir(exist_ok=True)
torch.save({"state_dict": best_state, "in_dim": F, "n_channels": C, "best_val": best_val}, save_path)
print(f"Saved to {save_path}", flush=True)

# ─── Routing test ───
print("\n=== Routing test ===", flush=True)
names = ["Instruct", "Reasoner", "Coder", "Tool", "Retrieval"]
label_map = {'translate': 0, 'explain': 0, 'reasoner': 1, 'coder': 2, 'tool': 3, 'retrieval': 4}
test_prompts = {
    'translate': ["translate", "dog", "to", "french"],
    'explain': ["explain", "gravity"],
    'reasoner': ["E0001", "is", "larger", "than", "E0002", ".", "E0002", "is", "larger", "than",
                 "E0003", ".", "Therefore", ",", "E0001", "is", "larger", "than"],
    'coder': ["def", "get_sum", "(", "a", ",", "b", ")", ":", "return", "a", "+"],
    'tool': ["What", "is", "54", "+", "23", "?"],
    'retrieval': ["What", "is", "a", "dog"],
}
for label, prompt in test_prompts.items():
    ids = torch.tensor([tok2id.get(t, tok2id["<UNK>"]) for t in prompt], device=DEVICE)
    T = len(prompt)
    p_outputs = [c.forward(ids).to(DEVICE) for c in channels]
    all_feats = []
    for t in range(T):
        feat = build_features_with_sigs(p_outputs, ids, t)
        all_feats.append(feat)
    feats = torch.stack(all_feats).to(DEVICE)
    with torch.no_grad():
        log_w = blender(feats.unsqueeze(0)).squeeze(0)
    w = log_w[-1].exp().tolist()
    best_ch = names[w.index(max(w))]
    expected = names[label_map[label]]
    ok = "✓" if best_ch == expected else "✗"
    w_str = " ".join(f"{n}={w[i]:.2f}" for i, n in enumerate(names))
    print(f"  {ok} {label:10s} → {best_ch:10s} [{w_str}]", flush=True)
