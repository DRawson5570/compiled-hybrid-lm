"""CMI Blender — Classification training (not NLL).

Train the causal conv to predict WHICH CHANNEL should fire, given the feature
signatures. This is a supervised classification problem, not an NLL problem.
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hybrid.v2_capabilities.channels import InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
from hybrid.v2_capabilities.retrieval_channel import RetrievalChannel, DEFAULT_DOCS
from hybrid.v2_capabilities.dataset import tok2id, id2tok, V, get_ppmi_embeddings
from hybrid.v3_super_blender.model import CausalConvBlender

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
torch.manual_seed(42)
random.seed(42)

emb = get_ppmi_embeddings().to(DEVICE)
d = emb.shape[1]

# ─── Signature token IDs ───
SIG_TOKENS = {
    'instruct': ['chien', 'chat', 'pomme', 'force', 'mass', 'earth', 'attraction'],
    'reasoner': ['E0001', 'E0002', 'E0003', 'Therefore', 'larger', 'than'],
    'coder': ['def', 'return', 'import', 'numpy', 'as', 'np', 'zeros', '+', ':', '('],
    'tool': ['[USE_TOOL:', 'calculator', 'expr=', '54+23', '12+15', '77', '27'],
    'retrieval': ['dog', 'cat', 'gravity', 'earth', 'french'],
}
sig_ids = {k: torch.tensor([tok2id[t] for t in v if tok2id.get(t) is not None], device=DEVICE)
           for k, v in SIG_TOKENS.items()}


def build_features(p_outputs, ids, t):
    x_o = ids[t]
    # Distribution stats per channel
    logp_o = torch.stack([p[t, x_o] for p in p_outputs])
    ent = torch.stack([-(p[t].exp() * p[t]).sum() for p in p_outputs])

    # Channel signatures: avg log-prob of signature tokens
    sigs = torch.zeros(5, device=DEVICE)
    for ci, sn in enumerate(['instruct', 'reasoner', 'coder', 'tool', 'retrieval']):
        tids = sig_ids[sn]
        if len(tids) > 0:
            sigs[ci] = p_outputs[ci][t, tids].mean()

    # Embedding
    emb_feat = emb[x_o]

    # Key innovation: per-channel activation signals
    # For most channels, the default output is -5.1 (flat log-prob ≈ log(1/V)).
    # A "spike" above this means that channel has a strong opinion.
    # The signature values that are far from -5.1 indicate channel activity.
    spike = torch.stack([logp_o.max() for _ in range(1)])  # placeholder

    return torch.cat([logp_o, ent, sigs, emb_feat])


# ─── Build channels ───
print("Building channels...", flush=True)
channels = [
    InstructChannel(tok2id, id2tok, emb),
    ReasonerChannel(tok2id, id2tok),
    CoderChannel(tok2id, id2tok),
    ToolChannel(tok2id, id2tok),
    RetrievalChannel(tok2id, id2tok, emb, doc_texts=DEFAULT_DOCS),
]
C = len(channels)
F = 2 * C + C + d  # logp_o(C) + ent(C) + sigs(C) + emb(d) = 10+5+16 = 31
print(f"C={C}, F={F}", flush=True)

# ─── Generate training prompts with channel labels ───
prompt_data = {
    0: [["translate", "dog", "to", "french"],
        ["translate", "cat", "to", "french"],
        ["translate", "apple", "to", "french"],
        ["explain", "gravity"],
        ["explain", "dog"]] * 40,
    1: [["E0001", "is", "larger", "than", "E0002", ".",
         "E0002", "is", "larger", "than", "E0003", ".",
         "Therefore", ",", "E0001", "is", "larger", "than"]] * 60,
    2: [["def", "get_sum", "(", "a", ",", "b", ")", ":", "return", "a", "+", "b"],
        ["import", "numpy", "as", "np"],
        ["np", ".", "zeros", "(", "10", ")"]] * 60,
    3: [["What", "is", "54", "+", "23", "?"],
        ["What", "is", "12", "+", "15", "?"],
        ["54", "+", "23", "[USE_TOOL:", "calculator", "expr=", "54+23", "]", "Answer", "is", "77"]] * 60,
    4: [["What", "is", "a", "dog"],
        ["What", "is", "a", "cat"],
        ["What", "is", "gravity"]] * 60,
}

all_prompts, all_labels = [], []
for ch, prompts in prompt_data.items():
    for p in prompts:
        all_prompts.append(p)
        all_labels.append(ch)
print(f"{len(all_prompts)} training sequences", flush=True)

# ─── Build feature vectors ───
print("Building features...", flush=True)
feat_list, label_list = [], []
for prompt, label in zip(all_prompts, all_labels):
    ids = torch.tensor([tok2id.get(t, tok2id["<UNK>"]) for t in prompt], device=DEVICE)
    p_outputs = [c.forward(ids).to(DEVICE) for c in channels]
    # Use the LAST position's features (fully-formed prompt context)
    feat = build_features(p_outputs, ids, len(prompt) - 1)
    feat_list.append(feat)
    label_list.append(label)

features = torch.stack(feat_list).to(DEVICE)
labels = torch.tensor(label_list, dtype=torch.long, device=DEVICE)
N = features.shape[0]
print(f"{N} examples", flush=True)

# ─── Train as classifier ───
print("Training classifier...", flush=True)
blender = CausalConvBlender(in_dim=F, n_channels=C, channels=32, kernel_size=3).to(DEVICE)
for p in blender.parameters():
    if p.dim() >= 2:
        nn.init.xavier_uniform_(p)

opt = torch.optim.AdamW(blender.parameters(), lr=1e-3, weight_decay=1e-5)

n_train = int(N * 0.8)
perm = torch.randperm(N)
f_tr, l_tr = features[perm[:n_train]], labels[perm[:n_train]]
f_va, l_va = features[perm[n_train:]], labels[perm[n_train:]]

best_acc, best_ep = 0.0, 0
best_state = None

for ep in range(1500):
    blender.train()
    for s in range(0, n_train, 128):
        j = torch.arange(s, min(s + 128, n_train), device=DEVICE)
        # Causal conv expects (B, T, F) — wrap each example as a 1-timestep sequence
        log_w = blender(f_tr[j].unsqueeze(1)).squeeze(1)  # (B, 1, C) -> (B, C)
        loss = F.cross_entropy(log_w, l_tr[j])
        opt.zero_grad()
        loss.backward()
        opt.step()

    blender.eval()
    with torch.no_grad():
        log_w = blender(f_va.unsqueeze(1)).squeeze(1)
        val_loss = F.cross_entropy(log_w, l_va).item()
        preds = log_w.argmax(dim=1)
        acc = (preds == l_va).float().mean().item()

    if acc > best_acc:
        best_acc = acc
        best_ep = ep
        best_state = {k: v.clone() for k, v in blender.state_dict().items()}

    if ep % 100 == 0:
        print(f"  ep {ep:4d}: val_loss={val_loss:.4f} acc={acc:.1%} best_acc={best_acc:.1%}@{best_ep}",
              flush=True)

# Save
blender.load_state_dict(best_state)
save_path = ROOT / "artifacts" / "blender_5ch_classifier.pt"
save_path.parent.mkdir(exist_ok=True)
torch.save({"state_dict": best_state, "in_dim": F, "n_channels": C, "best_acc": best_acc}, save_path)
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
    p_outputs = [c.forward(ids).to(DEVICE) for c in channels]
    feat = build_features(p_outputs, ids, len(prompt) - 1)
    with torch.no_grad():
        log_w = blender(feat.unsqueeze(0).unsqueeze(0)).squeeze()
    w = log_w.softmax(dim=0).tolist()
    best_ch = names[w.index(max(w))]
    expected = names[label_map[label]]
    ok = "✓" if best_ch == expected else "✗"
    w_str = " ".join(f"{n}={w[i]:.2f}" for i, n in enumerate(names))
    print(f"  {ok} {label:10s} → {best_ch:10s} [{w_str}]", flush=True)
