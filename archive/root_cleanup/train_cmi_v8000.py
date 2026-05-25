"""Full CMI training pipeline — V=8000 scale.

1. Generates synthetic prompts per capability using wikitext BPE tokens
2. Computes REAL channel features using compiled LM outputs (bigram, PPMI, shape, freq, recency)
3. Trains a classifier blender to route to the correct channel
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, random
from pathlib import Path
import sys

REPO = Path("/home/drawson/llm_decoupling")
sys.path.insert(0, str(REPO))
DS = Path("/home/drawson/deepseek_experiments")

from compile_wiki_lm_v13 import load_setup, load_or_build_tokens, DEVICE


def build_all_channels(bpe, ids_np, emb, V):
    """Build all 5 compiled channels for V=8000. Returns list of (V, V) log-prob matrices."""
    print("  Building channels...", flush=True)
    emb_norm = F.normalize(emb.float(), dim=1)

    # 1. PPMI semantic (Instruct)
    sims = emb_norm @ emb_norm.T * 50.0
    ppmi_lp = F.log_softmax(sims, dim=1)  # (V, V)

    # 2. Bigram transition (Reasoner)
    bigram = np.zeros((V, V), dtype=np.float64)
    for t in range(min(len(ids_np) - 1, 5_000_000)):
        bigram[ids_np[t], ids_np[t + 1]] += 1
    a = 0.01
    bigram_p = (bigram + a) / (bigram.sum(axis=1, keepdims=True) + a * V)
    bigram_lp = torch.from_numpy(np.log(bigram_p.clip(1e-30))).float()

    # 3. Shape/syntax (Coder)
    shape_map = np.zeros(V, dtype=np.int32)
    for tid in range(V):
        s = bpe.decode([tid]).strip().lower()
        if not s: shape_map[tid] = 6
        elif s.isdigit(): shape_map[tid] = 4
        elif not any(c.isalpha() for c in s): shape_map[tid] = 5
        elif s.islower(): shape_map[tid] = 0
        elif s.isupper(): shape_map[tid] = 1
        elif s[0].isupper(): shape_map[tid] = 2
        else: shape_map[tid] = 3
    shape_lp = torch.full((V, V), -float("inf"))
    for tid in range(V):
        s = shape_map[tid]
        same = torch.from_numpy(shape_map == s)
        n = same.sum().item()
        if n > 0:
            shape_lp[tid, same] = -math.log(n)

    # 4. Unigram frequency (Tool)
    cnts = np.bincount(ids_np[:5_000_000], minlength=V).astype(np.float64)
    p = (cnts + a) / (cnts.sum() + a * V)
    freq_lp = torch.from_numpy(np.log(p.clip(1e-30))).float().unsqueeze(0).expand(V, V)

    # 5. Recency-weighted bigram (Retrieval)
    window = 5000
    recent = np.zeros((V, V), dtype=np.float64)
    start = max(0, len(ids_np) - window)
    for t in range(start, len(ids_np) - 1):
        recent[ids_np[t], ids_np[t + 1]] += 1
    recent_p = (recent + a) / (recent.sum(axis=1, keepdims=True) + a * V)
    recency_lp = torch.from_numpy(np.log(recent_p.clip(1e-30))).float()

    return [ppmi_lp, bigram_lp, shape_lp, freq_lp, recency_lp], ['PPMI', 'Bigram', 'Shape', 'Freq', 'Recency']


def generate_prompts(bpe, tok2id, n_per_class=500, prompt_len=8):
    """Generate synthetic prompts for each capability type using real BPE tokens."""
    # Token classes for prompt generation
    common_ids = []
    entity_ids = []
    code_ids = []
    number_ids = []
    punct_ids = []

    for tid in range(len(tok2id)):
        s = bpe.decode([tid]).strip()
        if not s: continue
        common_ids.append(tid)
        if s[0].isupper() and len(s) > 1: entity_ids.append(tid)
        if s.isdigit(): number_ids.append(tid)
        if s in {'+', '-', '*', '/', '=', '<', '>', '?', ':', 'def', 'return', 'import'}:
            code_ids.append(tid)
            punct_ids.append(tid)

    # Fill up if too few
    if len(code_ids) < 10: code_ids = number_ids + punct_ids
    if len(entity_ids) < 10: entity_ids = common_ids
    if len(number_ids) < 10: number_ids = common_ids
    if len(punct_ids) < 5: punct_ids = common_ids

    prompts = []
    labels = []

    # Instruct: common + entity tokens
    for _ in range(n_per_class):
        p = random.choices(common_ids[:500], k=prompt_len)
        prompts.append(p)
        labels.append(0)

    # Reasoner: entity-heavy
    for _ in range(n_per_class):
        p = random.choices(entity_ids[:200] + common_ids[:200], k=prompt_len)
        prompts.append(p)
        labels.append(1)

    # Coder: code + punct tokens
    for _ in range(n_per_class):
        p = random.choices(code_ids[:30] + punct_ids[:30] + common_ids[:100], k=prompt_len)
        prompts.append(p)
        labels.append(2)

    # Tool: number + operator sequences
    for _ in range(n_per_class):
        p = random.choices(number_ids[:50] + punct_ids[:30], k=prompt_len)
        prompts.append(p)
        labels.append(3)

    # Retrieval: common tokens
    for _ in range(n_per_class):
        p = random.choices(common_ids[:500], k=prompt_len)
        prompts.append(p)
        labels.append(4)

    return prompts, labels


def compute_features(channels_lp, prompts, labels, emb, d):
    """Compute feature vectors from real channel outputs."""
    C = len(channels_lp)
    F = 4 * C + d
    X, y = [], []

    # Move channels to same device as emb
    ch_lp = [lp.to(emb.device) for lp in channels_lp]

    for prompt, label in zip(prompts, labels):
        ids_t = torch.tensor(prompt, device=emb.device).long()
        t = len(prompt) - 1
        x_o = ids_t[t]
        x_l1 = ids_t[t - 1] if t > 0 else torch.zeros_like(ids_t[t])

        feat_parts = []
        for ch in range(C):
            feat_parts.append(ch_lp[ch][x_o, x_o].unsqueeze(0))
        for ch in range(C):
            feat_parts.append(ch_lp[ch][x_l1, x_l1].unsqueeze(0))
        for ch in range(C):
            lp = ch_lp[ch][x_o]
            p = lp.exp()
            mask = (lp > -float("inf")) & (~torch.isnan(lp))
            entropy = -(p[mask] * lp[mask]).sum() if mask.any() else torch.tensor(0.0, device=emb.device)
            feat_parts.append(entropy.unsqueeze(0))
        for ch in range(C):
            feat_parts.append(ch_lp[ch][x_o].max().unsqueeze(0))
        feat_parts.append(emb[x_o])

        feat_cat = torch.cat(feat_parts)
        # Ensure absolutely NO NaNs sneak into features
        feat_cat = torch.nan_to_num(feat_cat, nan=0.0)
        X.append(feat_cat.cpu().numpy())
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class SimpleClassifier(nn.Module):
    """Simple MLP classifier for channel routing."""
    def __init__(self, in_dim, n_classes, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def main():
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    emb = emb.to(DEVICE)
    ids_all = load_or_build_tokens(bpe, bpe_to_lm, V)
    ids_np = ids_all.numpy()

    # Build channels
    t0 = time.time()
    channels_lp, ch_names = build_all_channels(bpe, ids_np, emb, V)
    print(f"  Channels built in {time.time()-t0:.0f}s", flush=True)

    # Generate training data
    print("Generating prompts...", flush=True)
    prompts, labels = generate_prompts(bpe, tok2id, n_per_class=1500, prompt_len=8)
    print(f"  {len(prompts)} prompts", flush=True)

    # Compute features
    print("Computing features...", flush=True)
    X, y = compute_features(channels_lp, prompts, labels, emb, d)
    input_dim = X.shape[1]
    print(f"  {len(X)} examples, F={input_dim}", flush=True)
    for ci, name in enumerate(ch_names):
        print(f"    {name}: {(y == ci).sum()}", flush=True)

    # Save
    np.savez(DS / "artifacts/cmi_v8000_train.npz", X=X, y=y, F=input_dim)
    print(f"Saved to artifacts/cmi_v8000_train.npz", flush=True)

    # Train classifier
    print("\nTraining classifier...", flush=True)
    torch.manual_seed(42)
    X_t = torch.from_numpy(X).float().to(DEVICE)
    y_t = torch.from_numpy(y).long().to(DEVICE)
    C = len(ch_names)

    n_train = int(len(X) * 0.8)
    perm = torch.randperm(len(X))
    X_tr, y_tr = X_t[perm[:n_train]], y_t[perm[:n_train]]
    X_va, y_va = X_t[perm[n_train:]], y_t[perm[n_train:]]

    model = SimpleClassifier(input_dim, C, hidden=128).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_acc, best_state = 0.0, None

    for ep in range(500):
        model.train()
        for s in range(0, n_train, 256):
            j = torch.arange(s, min(s + 256, n_train), device=DEVICE)
            logits = model(X_tr[j])
            loss = F.cross_entropy(logits, y_tr[j])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_va).argmax(dim=1)
            acc = (preds == y_va).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 100 == 0:
            print(f"  ep {ep:3d}: val_acc={acc:.1%} best={best_acc:.1%}", flush=True)

    print(f"\nBest val accuracy: {best_acc:.1%}", flush=True)

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "in_dim": input_dim, "n_classes": C, "acc": best_acc},
               DS / "artifacts/cmi_v8000_classifier.pt")
    print(f"Saved classifier", flush=True)

    # Quick test
    print(f"\nPer-class accuracy:", flush=True)
    with torch.no_grad():
        preds_all = model(X_va).argmax(dim=1)
    for ci, name in enumerate(ch_names):
        mask = y_va == ci
        if mask.sum() > 0:
            cls_acc = (preds_all[mask] == ci).float().mean().item()
            print(f"  {name}: {cls_acc:.1%}", flush=True)


if __name__ == "__main__":
    main()
