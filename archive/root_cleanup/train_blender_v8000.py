"""train_blender_v8000.py

Compiles 5 next-token channels on V=8000 wikitext, builds sequence-aware features,
and trains 4 dynamic mixture-of-experts neural blenders:
1. WindowMLPBlender
2. LookbackMLPBlender
3. GRUBlender
4. CausalConvBlender

Uses vectorized, robust feature construction and optimizes the mixture negative log-likelihood (Mixture NLL).
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys, time, math, json
from pathlib import Path

# Fix python import paths specifically for our local workspace
sys.path.insert(0, "/home/drawson/deepseek_experiments")

from compile_wiki_lm_v13 import load_setup, load_or_build_tokens, DEVICE
# Force deepseek_experiments folder into path before loading downstream
from train_cmi_v8000 import build_all_channels

# Re-insert our local workspace on top after compile_wiki_lm_v13 has prepended its repo path
sys.path.insert(0, "/home/drawson/deepseek_experiments")
DS = Path("/home/drawson/deepseek_experiments")

from hybrid.v3_super_blender.model import WindowMLPBlender, LookbackMLPBlender, GRUBlender, CausalConvBlender
from hybrid.v1_blender.blender_model import mixture_nll


def build_vectorized_features(ids, channels_lp, emb, target_device):
    """
    Constructs sequence-aware features and log_p_targets over a sequence of ids.
    Performs heavy computations on CPU to conserve CUDA VRAM, then moves the
    final highly compressed feature representations to target_device.
    ids: tensor of shape (N + 1,)
    x_o sequence is ids[0:N]
    y_t sequence is ids[1:N+1]
    """
    N = len(ids) - 1
    C = len(channels_lp)
    V, d = emb.shape

    # Ensure everything used for feature calculation is on CPU
    ids_cpu = ids.cpu()
    emb_cpu = emb.cpu()
    channels_cpu = [lp.cpu() for lp in channels_lp]

    x_o = ids_cpu[0:N]
    targets = ids_cpu[1:N+1]
    x_l1 = torch.zeros_like(x_o)
    x_l1[1:] = x_o[0:N-1]

    log_p_targets = torch.zeros((N, C), device="cpu")
    log_p_observed = torch.zeros((N, C), device="cpu")
    log_p_lag1 = torch.zeros((N, C), device="cpu")
    entropy = torch.zeros((N, C), device="cpu")
    max_log_prob = torch.zeros((N, C), device="cpu")

    for c in range(C):
        lp_matrix = channels_cpu[c] # (V, V) on CPU
        # Slice for all active current observed ids
        lp_slices = lp_matrix[x_o] # (N, V)
        
        # 1. Target log prob
        log_p_targets[:, c] = lp_slices[torch.arange(N), targets]

        # 2. Observed log prob
        log_p_observed[:, c] = lp_slices[torch.arange(N), x_o]

        # 3. Lag 1 log prob
        lp_lag_slices = lp_matrix[x_l1]
        log_p_lag1[:, c] = lp_lag_slices[torch.arange(N), x_l1]

        # 4. Max log prob
        max_log_prob[:, c] = lp_slices.max(dim=1).values

        # 5. Secure Entropy
        probs = lp_slices.exp()
        mask = lp_slices > -1e9
        term = probs * lp_slices
        term = torch.where(mask, term, torch.zeros_like(term))
        entropy[:, c] = -term.sum(dim=1)

    emb_feats = emb_cpu[x_o] # (N, d)
    features = torch.cat([log_p_observed, log_p_lag1, entropy, max_log_prob, emb_feats], dim=1)
    features = torch.nan_to_num(features, nan=0.0)
    
    # Move final compressed small outputs to CUDA target device
    return features.to(target_device), log_p_targets.to(target_device), targets.to(target_device)


def train_and_eval_blenders():
    bpe, vocab, tok2id, bpe_to_lm, emb, V, d = load_setup()
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = emb.to(target_device)
    print(f"Loaded setups: V={V}, d={d}")

    # Load corpus tokens
    ids_all = load_or_build_tokens(bpe, bpe_to_lm, V).to(target_device)
    ids_np = ids_all.cpu().numpy()

    # Build/load channels
    print("Compiling all next-token expert channels...")
    channels_lp, ch_names = build_all_channels(bpe, ids_np, emb, V)
    channels_lp = [lp.to(target_device) for lp in channels_lp]
    C = len(channels_lp)
    F_dim = 4 * C + d
    print(f"Channels ready. Single-step feature dim: {F_dim}")

    # Segment sequential split of corpus to train blenders
    # Let's take the last 50,000 tokens for training and next 10,000 for validation
    train_start = len(ids_all) - 80_000
    train_end = len(ids_all) - 10_000
    val_start = len(ids_all) - 10_000
    val_end = len(ids_all)

    print(f"Building vectorized features for train set ({train_end - train_start} tokens)...")
    train_feats, train_lpt, train_targets = build_vectorized_features(
        ids_all[train_start:train_end+1], channels_lp, emb, target_device
    )
    
    print(f"Building vectorized features for validation set ({val_end - val_start} tokens)...")
    val_feats, val_lpt, val_targets = build_vectorized_features(
        ids_all[val_start:val_end+1], channels_lp, emb, target_device
    )

    models_dir = DS / "artifacts/compiled_wiki_lm_v8000_blenders"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Define the 4 blenders to train
    blenders = {
        "window_mlp": WindowMLPBlender(
            single_step_dim=F_dim, n_channels=C, lookback_window=4, hidden=128
        ).to(target_device),
        "lookback_mlp": LookbackMLPBlender(
            single_step_dim=F_dim, n_channels=C, lookback_window=4, hidden=128, num_layers=2
        ).to(target_device),
        "gru": GRUBlender(
            in_dim=F_dim, n_channels=C, hidden=128, num_layers=2
        ).to(target_device),
        "causal_conv": CausalConvBlender(
            in_dim=F_dim, n_channels=C, channels=128, kernel_size=3, num_layers=3
        ).to(target_device),
    }

    # Helper evaluation function
    def eval_blender(name, model):
        model.eval()
        with torch.no_grad():
            if name in ["window_mlp", "lookback_mlp"]:
                log_w = model(val_feats, is_already_windowed=False)
            elif name == "gru":
                log_w, _ = model(val_feats.unsqueeze(0))
                log_w = log_w.squeeze(0)
            elif name == "causal_conv":
                log_w = model(val_feats.unsqueeze(0)).squeeze(0)
            
            # Loss and perplexity
            loss = mixture_nll(log_w, val_lpt)
            avg_nll = loss.mean().item()
            ppl = math.exp(avg_nll)

            # Accuracy (argmax channel predictions)
            # Reconstruct blended probabilities to find top-1 next-token prediction
            # blended_probs = sum(w[c] * p_c)
            # For speed, let's do a sub-sampled top-1 metric over 1,000 steps
            sample_size = min(1000, len(val_targets))
            correct = 0
            w_weights = log_w.exp()
            for idx in range(sample_size):
                w_idx = w_weights[idx]
                y_tgt = val_targets[idx].item()
                
                blended_prob = torch.zeros(V, device=target_device)
                x_o = ids_all[val_start + idx]
                for c_idx in range(C):
                    blended_prob += w_idx[c_idx] * channels_lp[c_idx][x_o].exp()
                if blended_prob.argmax().item() == y_tgt:
                    correct += 1
            acc_pct = (correct / sample_size) * 100.0 if sample_size > 0 else 0.0

        return ppl, acc_pct

    results = {}

    for name, model in blenders.items():
        print(f"\n--- Training {name.upper()} Blender ---")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        epochs = 15
        batch_size = 256
        n_train = len(train_feats)
        best_ppl = float("inf")
        best_state = None

        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            
            if name in ["gru", "causal_conv"]:
                # Sequence-level steps, process in chunks/batches
                # We can chunk the train sequence into mini-sequences of length 1024
                seq_len = 1024
                for s in range(0, n_train, seq_len):
                    end_idx = min(s + seq_len, n_train)
                    sub_feats = train_feats[s:end_idx].unsqueeze(0) # (1, T, F)
                    sub_lpt = train_lpt[s:end_idx] # (T, C)

                    opt.zero_grad()
                    if name == "gru":
                        log_w, _ = model(sub_feats)
                        log_w = log_w.squeeze(0)
                    else:
                        log_w = model(sub_feats).squeeze(0)

                    loss = mixture_nll(log_w, sub_lpt).mean()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item() * (end_idx - s)
                avg_loss = total_loss / n_train
            else:
                # MLP sequence processing
                perm = torch.randperm(n_train)
                # Build windowed features beforehand to optimize epoch training speed for LookbackMLP/WindowMLP
                with torch.no_grad():
                    windowed_train_feats = model.build_windowed_features(train_feats)

                for s in range(0, n_train, batch_size):
                    j = perm[s:min(s + batch_size, n_train)]
                    sub_feats = windowed_train_feats[j]
                    sub_lpt = train_lpt[j]

                    opt.zero_grad()
                    log_w = model(sub_feats, is_already_windowed=True)
                    loss = mixture_nll(log_w, sub_lpt).mean()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item() * len(j)
                avg_loss = total_loss / n_train

            val_ppl, val_acc = eval_blender(name, model)
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if ep % 5 == 0 or ep == epochs - 1:
                print(f"  Epoch {ep:2d}/{epochs} | Train Loss: {avg_loss:.4f} | Val PPL: {val_ppl:.2f} | Val Acc: {val_acc:.1f}%")

        print(f"  Best Val Perplexity: {best_ppl:.2f}")
        model.load_state_dict(best_state)
        torch.save(best_state, models_dir / f"blender_{name}.pt")
        results[name] = {"val_ppl": best_ppl, "val_acc": val_acc}

    # Save summary report
    summary_path = models_dir / "blender_report.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all blenders and reports to {models_dir}")


if __name__ == "__main__":
    train_and_eval_blenders()
