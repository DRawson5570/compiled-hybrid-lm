"""CMI Blender Training — Rich synthetic dataset + causal conv blender.

Generates enough diverse per-channel examples that the causal conv MUST learn
distinct routing patterns rather than collapsing to one channel.
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hybrid.v2_capabilities.channels import InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
from hybrid.v2_capabilities.retrieval_channel import RetrievalChannel, DEFAULT_DOCS
from hybrid.v2_capabilities.dataset import tok2id, id2tok, V, emb_dim, get_ppmi_embeddings, VOCAB_WORDS
from hybrid.v1_blender.blender_model import build_feature_matrix, mixture_nll
from hybrid.v3_super_blender.model import CausalConvBlender

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}", flush=True)
torch.manual_seed(42)
random.seed(42)

emb = get_ppmi_embeddings().to(DEVICE)

# ─── Generate per-channel training sequences ───
# Each channel gets a UNIQUE token embedding signature so the blender can
# distinguish them by feature statistics alone
#
# Strategy: build prompt+tgt pairs where each channel is the CLEAR winner
# for specific prompts. The blender sees 5 different kinds of feature patterns
# and learns to route accordingly.

def build_vocab_subset(tokens: list[str], n: int) -> list[list[str]]:
    """Generate random prompt sequences from a subset of the vocabulary."""
    results = []
    for _ in range(n):
        length = random.randint(2, 6)
        seq = random.choices(tokens, k=length)
        results.append(seq)
    return results

# Instruct prompts: translation/explanatory tokens
instruct_tokens = ["translate", "explain", "dog", "cat", "apple", "gravity",
                   "to", "french", "force", "mass", "earth", "attraction", ".", ","]
instruct_prompts = build_vocab_subset(instruct_tokens, 200)

# Reasoner prompts: entity comparison tokens
reasoner_tokens = ["E0001", "E0002", "E0003", "E0004", "E0005", "E0006",
                   "is", "larger", "than", "Therefore", ","]
reasoner_prompts = build_vocab_subset(reasoner_tokens, 200)

# Coder prompts: programming tokens
coder_tokens = ["def", "return", "import", "numpy", "as", "np", "zeros",
                "get_sum", "a", "b", ":", "+", "(", ")", ",", ".", "10"]
coder_prompts = build_vocab_subset(coder_tokens, 200)

# Tool prompts: calculator tokens
tool_tokens = ["What", "is", "54", "23", "12", "15", "100", "200", "8", "9",
               "+", "?", "[USE_TOOL:", "calculator", "expr=", "]", "Answer", "77", "27", "300", "17"]
tool_prompts = build_vocab_subset(tool_tokens, 200)

# Retrieve prompts: document Q&A tokens
retrieve_tokens = ["What", "is", "a", "dog", "cat", "larger", "than",
                   "gravity", "force", "mass", "earth", "translate", "explain"]
retrieve_prompts = build_vocab_subset(retrieve_tokens, 200)

# Assign labels — each prompt gets a one-hot channel label
# (the blender target is the per-channel log-prob on the target token,
#  but we ALSO want the blender weights to match the intended channel)
all_sequences = []
all_labels = []  # which channel should dominate

for p in instruct_prompts:
    all_sequences.append(p)
    all_labels.append(0)  # InstructChannel
for p in reasoner_prompts:
    all_sequences.append(p)
    all_labels.append(1)  # ReasonerChannel
for p in coder_prompts:
    all_sequences.append(p)
    all_labels.append(2)  # CoderChannel
for p in tool_prompts:
    all_sequences.append(p)
    all_labels.append(3)  # ToolChannel
for p in retrieve_prompts:
    all_sequences.append(p)
    all_labels.append(4)  # RetrievalChannel

# Shuffle
combined = list(zip(all_sequences, all_labels))
random.shuffle(combined)
all_sequences, all_labels = zip(*combined)
all_sequences = list(all_sequences)
all_labels = list(all_labels)

print(f"{len(all_sequences)} total training sequences", flush=True)

# ─── Build features ───
print("Building channels...", flush=True)
channels = [
    InstructChannel(tok2id, id2tok, emb),
    ReasonerChannel(tok2id, id2tok),
    CoderChannel(tok2id, id2tok),
    ToolChannel(tok2id, id2tok),
    RetrievalChannel(tok2id, id2tok, emb, doc_texts=DEFAULT_DOCS),
]
C = len(channels)
F = 4 * C + emb_dim
print(f"C={C}, F={F}", flush=True)

print("Building features...", flush=True)
feat_list, logp_list, target_weights_list = [], [], []

for ctx, label in zip(all_sequences, all_labels):
    ids = torch.tensor([tok2id.get(t, tok2id["<UNK>"]) for t in ctx], device=DEVICE)
    T = len(ctx)
    p_outputs = [c.forward(ids).to(DEVICE) for c in channels]

    for t in range(T):
        x_o = ids[t]
        x_l1 = ids[t - 1] if t > 0 else torch.zeros_like(ids[t])
        logp_o = torch.stack([p[t, x_o] for p in p_outputs])
        logp_l1 = torch.stack([p[t, x_l1] for p in p_outputs])
        ent = torch.stack([-(p[t].exp() * p[t]).sum() for p in p_outputs])
        mlp = torch.stack([p[t].max() for p in p_outputs])
        feat = build_feature_matrix(
            logp_o.unsqueeze(0), logp_l1.unsqueeze(0),
            ent.unsqueeze(0), mlp.unsqueeze(0),
            emb.to(DEVICE), x_o.unsqueeze(0), use_embedding=True,
        )
        feat_list.append(feat)
        # Target: log-prob of the top token per channel (for NLL loss)
        tid = p_outputs[0][t].argmax().item()  # use channel 0's top token as target
        logp_tgt = torch.stack([p[t, tid] for p in p_outputs])
        logp_list.append(logp_tgt)

features = torch.cat(feat_list, dim=0).to(DEVICE)
log_p_targets = torch.stack(logp_list).to(DEVICE)
N = features.shape[0]
print(f"{N} training tokens", flush=True)

# ─── Train ───
print("Training...", flush=True)
blender = CausalConvBlender(in_dim=F, n_channels=C, channels=64, kernel_size=3).to(DEVICE)
# Xavier init (NOT zeros) so all channels start with non-trivial weights
for p in blender.parameters():
    if p.dim() >= 2:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.zeros_(p)

opt = torch.optim.AdamW(blender.parameters(), lr=3e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000)

n_train = int(N * 0.85)
perm = torch.randperm(N)
f_tr, f_va = features[perm[:n_train]], features[perm[n_train:]]
l_tr, l_va = log_p_targets[perm[:n_train]], log_p_targets[perm[n_train:]]
print(f"Train: {n_train}, Val: {N - n_train}", flush=True)

best_val, best_ep = float("inf"), 0
best_state = None
patience, no_improve = 400, 0

for ep in range(3000):
    blender.train()
    # Train in chunks of 64 to fit in GPU
    for s in range(0, n_train, 64):
        j = torch.arange(s, min(s + 64, n_train), device=DEVICE)
        log_w = blender(f_tr[j].unsqueeze(0)).squeeze(0)
        nll = mixture_nll(log_w, l_tr[j]).mean()

        # Entropy bonus: discourage collapse to single channel
        w_mean = log_w.exp().mean(dim=0)
        ent = -(w_mean * torch.log(w_mean + 1e-8)).sum()
        loss = nll - 0.001 * ent

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

# ─── Save ───
blender.load_state_dict(best_state)
save_dir = ROOT / "artifacts"
save_dir.mkdir(exist_ok=True)
save_path = save_dir / "blender_5ch_v2.pt"
torch.save({"state_dict": best_state, "in_dim": F, "n_channels": C,
             "best_val_nll": best_val, "best_ep": best_ep}, save_path)
print(f"\nSaved to {save_path}", flush=True)

# ─── Routing test ───
print("\n=== Routing test ===", flush=True)
names = ["Instruct", "Reasoner", "Coder", "Tool", "Retrieval"]
test_prompts = {
    "translate": ["translate", "dog", "to", "french"],
    "explain":   ["explain", "gravity"],
    "reasoner":  ["E0001", "is", "larger", "than", "E0002", ",",
                  "E0002", "is", "larger", "than", "E0003", ".",
                  "Therefore", ",", "E0001", "is", "larger", "than"],
    "coder":     ["def", "get_sum", "(", "a", ",", "b", ")", ":", "return", "a", "+"],
    "tool":      ["What", "is", "54", "+", "23", "?"],
    "retrieval": ["What", "is", "a", "dog"],
}

for label, prompt in test_prompts.items():
    ids = torch.tensor([tok2id.get(t, tok2id["<UNK>"]) for t in prompt], device=DEVICE)
    T = len(prompt)
    p_outputs = [c.forward(ids).to(DEVICE) for c in channels]

    all_feats = []
    for t in range(T):
        x_o, x_l1 = ids[t], ids[t - 1] if t > 0 else torch.zeros_like(ids[t])
        logp_o = torch.stack([p[t, x_o] for p in p_outputs])
        logp_l1 = torch.stack([p[t, x_l1] for p in p_outputs])
        ent = torch.stack([-(p[t].exp() * p[t]).sum() for p in p_outputs])
        mlp = torch.stack([p[t].max() for p in p_outputs])
        feat = build_feature_matrix(
            logp_o.unsqueeze(0), logp_l1.unsqueeze(0),
            ent.unsqueeze(0), mlp.unsqueeze(0),
            emb.to(DEVICE), x_o.unsqueeze(0), use_embedding=True,
        )
        all_feats.append(feat)

    feats_t = torch.cat(all_feats, dim=0).to(DEVICE)
    with torch.no_grad():
        log_w = blender(feats_t.unsqueeze(0)).squeeze(0)
    w = log_w[-1].exp().tolist()
    best_ch = names[w.index(max(w))]
    w_str = " ".join(f"{n}={w[i]:.2f}" for i, n in enumerate(names))
    marker = "✓" if names.index(best_ch) == ["instruct","reasoner","coder","tool","retrieval"].index(label) else "✗"
    print(f"  {marker} {label:10s}: → {best_ch:10s} [{w_str}]", flush=True)
