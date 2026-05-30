from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_features: int,
        l1_lambda: float = 1e-3,
        tied_decoder: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.l1_lambda = l1_lambda
        self.tied_decoder = tied_decoder

        self.encoder = nn.Linear(d_model, n_features, bias=True)
        self.decoder = nn.Linear(n_features, d_model, bias=True)

        self._init_weights()

        self.W_enc: torch.Tensor
        self.W_dec: torch.Tensor

    def _init_weights(self):
        scale = 0.1 / (self.d_model ** 0.5)
        nn.init.normal_(self.encoder.weight, std=scale)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, std=scale)
        nn.init.zeros_(self.decoder.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_pre = self.encoder(x)
        h = F.relu(h_pre)
        x_hat = self.decoder(h)

        mse = F.mse_loss(x_hat, x)
        l1 = h.mean()

        sparsity = (h > 0).float().mean()

        return x_hat, h, mse

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def get_features(self) -> Dict[int, torch.Tensor]:
        features = {}
        for i in range(self.n_features):
            w_dec_col = self.decoder.weight[:, i].detach()
            if w_dec_col.norm() > 1e-8:
                features[i] = w_dec_col
        return features

    @property
    def encoder_weight(self) -> torch.Tensor:
        return self.encoder.weight.data

    @property
    def decoder_weight(self) -> torch.Tensor:
        return self.decoder.weight.data

    def top_activating_features(
        self, x: torch.Tensor, k: int = 10
    ) -> torch.Tensor:
        h = self.encode(x)
        return torch.topk(h, k=min(k, h.shape[-1]), dim=-1)


def train_sae(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    steps: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, list]:
    sae.train()
    sae.to(device)

    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    history = {
        "total_loss": [],
        "mse_loss": [],
        "l1_loss": [],
        "sparsity": [],
    }

    activations = activations.to(device)

    for step in range(steps):
        idx = torch.randperm(activations.shape[0])[:batch_size]
        batch = activations[idx]

        optimizer.zero_grad(set_to_none=True)
        x_hat, h, mse = sae(batch)

        l1 = h.mean()
        total = mse + sae.l1_lambda * l1
        sparsity = (h > 0).float().mean()

        total.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["total_loss"].append(float(total.item()))
        history["mse_loss"].append(float(mse.item()))
        history["l1_loss"].append(float(l1.item()))
        history["sparsity"].append(float(sparsity.item()))

        if verbose and (step == 0 or (step + 1) % 200 == 0 or step == steps - 1):
            print(
                f"  SAE step {step + 1:5d}/{steps}  "
                f"loss={total.item():.4f}  mse={mse.item():.4f}  "
                f"l1={l1.item():.4f}  sp={sparsity.item():.3f}",
                flush=True,
            )

    return history


def normalize_decoder(sae: SparseAutoencoder):
    with torch.no_grad():
        for i in range(sae.n_features):
            norm = sae.decoder.weight[:, i].norm()
            if norm > 1e-8:
                sae.decoder.weight[:, i] /= norm
                sae.encoder.weight[i, :] *= norm


def train_sae_on_layer_activations(
    sae: SparseAutoencoder,
    collector,
    layer: int,
    texts: List[str],
    steps: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda",
    normalize: bool = True,
) -> Dict[str, list]:
    print(f"Collecting activations from layer {layer}...")
    residual = collector.collect_residual_stream(texts, max_length=128)
    acts = residual.get(layer)
    if acts is None:
        raise ValueError(f"No activations found for layer {layer}")

    acts_flat = acts.reshape(-1, acts.shape[-1]).to(dtype=torch.float32)

    print(
        f"  Got {acts_flat.shape[0]} activation vectors of dim {acts_flat.shape[1]}"
    )

    print(f"Training SAE on layer {layer} ({steps} steps)...")
    history = train_sae(
        sae,
        acts_flat,
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    if normalize:
        normalize_decoder(sae)

    return history


def train_sae_on_mlp_activations(
    sae: SparseAutoencoder,
    model,
    tokenizer,
    layer: int,
    texts: List[str],
    steps: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str = "cuda",
    normalize: bool = True,
) -> Dict[str, list]:
    from .mlp_decomposer import extract_mlp_activations

    print(f"Collecting MLP gate activations from layer {layer}...")
    gate_pre, gate_post, up_out = extract_mlp_activations(
        model, tokenizer, texts, layer, device=device
    )

    if gate_pre.shape[0] == 0:
        raise ValueError(f"No MLP activations found for layer {layer}")

    acts_flat = gate_pre.to(dtype=torch.float32)
    print(f"  Got {acts_flat.shape[0]} gate activation vectors of dim {acts_flat.shape[1]}")

    print(f"Training SAE on MLP layer {layer} ({steps} steps)...")
    history = train_sae(
        sae,
        acts_flat,
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    if normalize:
        normalize_decoder(sae)

    return history


def extract_feature_vectors(
    sae: SparseAutoencoder,
) -> Dict[int, torch.Tensor]:
    return sae.get_features()


def top_activating_tokens_for_feature(
    sae: SparseAutoencoder,
    collector,
    feature_idx: int,
    layer: int,
    texts: List[str],
    n_top: int = 20,
) -> List[Tuple[str, float]]:
    tokenizer = collector.tokenizer

    residual = collector.collect_residual_stream(texts, max_length=128)
    acts = residual.get(layer)
    if acts is None:
        return []

    encoded = []
    for text in texts:
        tokens = tokenizer.encode(text, return_tensors="pt")[0]
        encoded.append(tokens)
    all_token_ids = torch.cat(encoded, dim=0)

    sae.eval()
    acts_flat = acts.reshape(-1, acts.shape[-1]).to(
        dtype=torch.float32,
        device=sae.encoder.weight.device,
    )

    n_vecs = min(acts_flat.shape[0], all_token_ids.shape[0])
    acts_flat = acts_flat[:n_vecs]
    all_token_ids = all_token_ids[:n_vecs]

    with torch.no_grad():
        h = sae.encode(acts_flat)
        feature_acts = h[:, feature_idx]

    token_scores = []
    for i in range(n_vecs):
        score = float(feature_acts[i].item())
        if score > 0:
            tid = int(all_token_ids[i].item())
            token_str = tokenizer.decode([tid])
            token_scores.append((token_str, score, tid))

    token_scores.sort(key=lambda x: x[1], reverse=True)
    return [(t, s) for t, s, _ in token_scores[:n_top]]
