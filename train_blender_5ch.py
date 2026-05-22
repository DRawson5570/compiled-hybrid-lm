"""Train 5-channel CMI blender on pe3 with expanded synthetic dataset.

Key fix: generate enough diverse training data so the blender learns per-prompt
routing, not just collapsing to one channel.
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hybrid.v2_capabilities.channels import InstructChannel, ReasonerChannel, CoderChannel, ToolChannel
from hybrid.v2_capabilities.retrieval_channel import RetrievalChannel, DEFAULT_DOCS
from hybrid.v2_capabilities.dataset import tok2id, id2tok, V, emb_dim, get_ppmi_embeddings
from hybrid.v1_blender.blender_model import build_feature_matrix, mixture_nll
from hybrid.v3_super_blender.model import CausalConvBlender

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
emb = get_ppmi_embeddings().to(DEVICE)

# ─── Build expanded training dataset ───
# Each capability gets MANY distinct examples so the blender learns to route
def build_expanded_dataset():
    pairs = []

    # Instruct: translations (many examples)
    translations = [
        (["translate", "dog", "to", "french"], "chien"),
        (["translate", "cat", "to", "french"], "chat"),
        (["translate", "apple", "to", "french"], "pomme"),
        (["translate", "gravity", "to", "french"], "gravité"),
        (["translate", "chien", "to", "french"], "chien"),
        (["translate", "chat", "to", "french"], "chat"),
        (["translate", "pomme", "to", "french"], "pomme"),
        (["translate", "dog", "to", "french", "."], "chien"),
        (["translate", "cat", "to", "french", "."], "chat"),
    ]

    # Instruct: explanations (many examples)
    explanations = [
        (["explain", "gravity"], "force"),
        (["explain", "gravity"], "mass"),
        (["explain", "gravity"], "earth"),
        (["explain", "gravity"], "attraction"),
        (["explain", "gravity", "."], "force"),
        (["explain", "dog"], "attraction"),
        (["explain", "cat"], "force"),
    ]

    # Reasoner: transitive chains
    reasoner_examples = [
        (["E0001", "is", "larger", "than", "E0002", ".", "E0002", "is", "larger", "than",
          "E0003", ".", "Therefore", ",", "E0001", "is", "larger", "than"], "E0003"),
        (["E0004", "is", "larger", "than", "E0005", ".", "E0005", "is", "larger", "than",
          "E0006", ".", "Therefore", ",", "E0004", "is", "larger", "than"], "E0006"),
        (["E0007", "is", "larger", "than", "E0008", ".", "E0008", "is", "larger", "than",
          "E0009", ".", "Therefore", ",", "E0007", "is", "larger", "than"], "E0009"),
        (["E0002", "is", "larger", "than", "E0003", ".", "E0003", "is", "larger", "than",
          "E0004", ".", "Therefore", ",", "E0002", "is", "larger", "than"], "E0004"),
        (["E0005", "is", "larger", "than", "E0006", ".", "E0006", "is", "larger", "than",
          "E0007", ".", "Therefore", ",", "E0005", "is", "larger", "than"], "E0007"),
    ]

    # Coder: function definition
    coder_examples = [
        (["def", "get_sum", "(", "a", ",", "b", ")"], ":"),
        (["def", "get_sum", "(", "a", ",", "b", ")", ":"], "return"),
        (["def", "get_sum", "(", "a", ",", "b", ")", ":", "return"], "a"),
        (["def", "get_sum", "(", "a", ",", "b", ")", ":", "return", "a"], "+"),
        (["def", "get_sum", "(", "a", ",", "b", ")", ":", "return", "a", "+"], "b"),
        (["def", "get_sum", "(", "a", ",", "b", ")", ":", "return", "a", "+", "b"], "."),
        (["import"], "numpy"),
        (["import", "numpy"], "as"),
        (["import", "numpy", "as"], "np"),
        (["import", "numpy", "as", "np"], "."),
        (["import", "numpy", "as", "np", "."], "zeros"),
        (["import", "numpy", "as", "np", ".", "zeros"], "("),
        (["import", "numpy", "as", "np", ".", "zeros", "("], "10"),
        (["import", "numpy", "as", "np", ".", "zeros", "(", "10"], ")"),
    ]

    # Tool: calculator
    tool_examples = [
        (["What", "is", "54", "+", "23", "?"], "[USE_TOOL:"),
        (["What", "is", "54", "+", "23", "?", "[USE_TOOL:", "calculator"], "expr="),
        (["What", "is", "54", "+", "23", "?", "[USE_TOOL:", "calculator", "expr="], "54+23"),
        (["What", "is", "54", "+", "23", "?", "[USE_TOOL:", "calculator", "expr=", "54+23"], "]"),
        (["What", "is", "54", "+", "23", "?", "[USE_TOOL:", "calculator", "expr=", "54+23", "]",
          "Answer", "is"], "77"),
        (["What", "is", "12", "+", "15", "?", "[USE_TOOL:", "calculator", "expr=", "12+15", "]",
          "Answer", "is"], "27"),
        (["What", "is", "100", "+", "200", "?", "[USE_TOOL:", "calculator", "expr=", "100+200", "]",
          "Answer", "is"], "300"),
        (["What", "is", "8", "+", "9", "?", "[USE_TOOL:", "calculator", "expr=", "8+9", "]",
          "Answer", "is"], "17"),
    ]

    # Retrieval: facts from stored documents (using only in-vocab tokens)
    retrieval_examples = [
        (["What", "is", "a", "dog"], "larger"),
        (["What", "is", "a", "cat"], "larger"),
        (["What", "is", "gravity"], "force"),
        (["What", "is", "mass"], "earth"),
        (["dog", "is", "larger", "than"], "cat"),
        (["cat", "is", "larger", "than"], "dog"),
    ]

    all_examples = (translations + explanations + reasoner_examples +
                    coder_examples + tool_examples + retrieval_examples)

    duplicate_factor = 5
    for _ in range(duplicate_factor):
        for ctx, tgt in all_examples:
            pairs.append((ctx, tgt))

    return pairs


# ─── Build features ───
print("Building channels...", flush=True)
instruct = InstructChannel(tok2id, id2tok, emb)
reasoner = ReasonerChannel(tok2id, id2tok)
coder = CoderChannel(tok2id, id2tok)
tool = ToolChannel(tok2id, id2tok)
retrieval = RetrievalChannel(tok2id, id2tok, emb, doc_texts=DEFAULT_DOCS)
channels = [instruct, reasoner, coder, tool, retrieval]
C = len(channels)
F = 4 * C + emb_dim
print(f"C={C}, F={F}", flush=True)

pairs = build_expanded_dataset()
print(f"{len(pairs)} training pairs", flush=True)

print("Building features...", flush=True)
feat_list, logp_list = [], []
for ctx, tgt in pairs:
    ids = torch.tensor([tok2id[t] for t in ctx], device=DEVICE)
    tid = tok2id[tgt]
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
        logp_list.append(torch.stack([p[t, tid] for p in p_outputs]))

features = torch.cat(feat_list, dim=0).to(DEVICE)
log_p_targets = torch.stack(logp_list).to(DEVICE)
N = features.shape[0]
print(f"{N} training tokens", flush=True)

# ─── Train blender ───
print("Training...", flush=True)
blender = CausalConvBlender(in_dim=F, n_channels=C, channels=64, kernel_size=3).to(DEVICE)
for p in blender.parameters():
    if p.dim() >= 2:
        nn.init.xavier_uniform_(p)

opt = torch.optim.AdamW(blender.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)

# Train/val split
n_train = int(N * 0.8)
perm = torch.randperm(N)
f_tr, f_va = features[perm[:n_train]], features[perm[n_train:]]
l_tr, l_va = log_p_targets[perm[:n_train]], log_p_targets[perm[n_train:]]

best_val, best_ep = float("inf"), 0
best_state = None
patience = 200
no_improve = 0

for ep in range(2000):
    blender.train()
    for s in range(0, n_train, 32):
        j = torch.arange(s, min(s + 32, n_train), device=DEVICE)
        log_w = blender(f_tr[j].unsqueeze(0)).squeeze(0)
        nll = mixture_nll(log_w, l_tr[j]).mean()
        opt.zero_grad()
        nll.backward()
        opt.step()
    sched.step()

    blender.eval()
    with torch.no_grad():
        log_w = blender(f_va.unsqueeze(0)).squeeze(0)
        val_nll = mixture_nll(log_w, l_va).mean().item()
        avg_w = log_w[0].exp().tolist()

    if val_nll < best_val:
        best_val = val_nll
        best_ep = ep
        best_state = {k: v.clone() for k, v in blender.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if ep % 100 == 0:
        print(f"  ep {ep:4d}: val_NLL={val_nll:.4f}  w=[{', '.join(f'{w:.2f}' for w in avg_w)}]  "
              f"best={best_val:.4f}@{best_ep}", flush=True)

    if no_improve > patience:
        print(f"  Early stop at ep {ep}", flush=True)
        break

# ─── Save ───
blender.load_state_dict(best_state)
save_dir = ROOT / "artifacts"
save_dir.mkdir(exist_ok=True)
save_path = save_dir / "blender_5ch.pt"
torch.save({"state_dict": best_state, "in_dim": F, "n_channels": C,
             "best_val_nll": best_val, "best_ep": best_ep}, save_path)
print(f"\nSaved to {save_path}", flush=True)

# ─── Test routing on different prompt types ───
print("\n=== Routing test ===", flush=True)
test_prompts = [
    ["translate", "dog", "to", "french"],
    ["explain", "gravity"],
    ["E0001", "is", "larger", "than", "E0002", ".", "E0002", "is", "larger", "than",
     "E0003", ".", "Therefore", ",", "E0001", "is", "larger", "than"],
    ["def", "get_sum", "(", "a", ",", "b", ")", ":", "return"],
    ["What", "is", "54", "+", "23", "?"],
    ["What", "is", "a", "dog"],
]

names = ["Instruct", "Reasoner", "Coder", "Tool", "Retrieval"]

for prompt in test_prompts:
    ids = torch.tensor([tok2id[t] for t in prompt], device=DEVICE)
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

    features_t = torch.cat(all_feats, dim=0).to(DEVICE)
    with torch.no_grad():
        log_w = blender(features_t.unsqueeze(0)).squeeze(0)
    w = log_w[-1].exp().tolist()
    best_ch = names[w.index(max(w))]
    w_str = " ".join(f"{names[i]}={w[i]:.2f}" for i in range(C))
    print(f"  {' '.join(prompt):45s} → {best_ch:10s}  [{w_str}]", flush=True)
