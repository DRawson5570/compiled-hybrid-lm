from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def extract_mlp_keys_values(
    model,
    layer: int,
    method: str = "neurons",
    n_clusters: int = 1024,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Decompose a Qwen2.5 gated MLP layer into key-value pairs.

    Args:
        model: Qwen2.5 model (loaded with transformers)
        layer: layer index (0..27)
        method: 'neurons' (8960 per-neuron pairs), 'clustered' (k-means grouping)
        n_clusters: number of clusters when method='clustered'
        seed: random seed for k-means

    Returns:
        {"keys": Tensor[N, d_model], "values": Tensor[N, d_model],
         "method": str, "n_neurons": int, "n_keys": int}
    """
    mlp = model.model.layers[layer].mlp

    gate_w = mlp.gate_proj.weight.data.clone().float().cpu()
    up_w = mlp.up_proj.weight.data.clone().float().cpu()
    down_w = mlp.down_proj.weight.data.clone().float().cpu()

    d_model = gate_w.shape[1]
    n_neurons = gate_w.shape[0]

    if method == "neurons" or (method == "auto" and n_neurons <= 2048):
        keys = gate_w.clone()
        values = down_w.t().clone()
        method_used = "neurons"
        n_keys = n_neurons

    elif method == "clustered" or method == "auto":
        k = max(2, min(n_clusters, n_neurons // 4))
        keys, values, assignments = _cluster_mlp(gate_w, up_w, down_w, k, seed)
        method_used = f"clustered_k{k}"
        n_keys = k

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "keys": keys,
        "values": values,
        "method": method_used,
        "n_neurons": n_neurons,
        "n_keys": n_keys,
    }


def _cluster_mlp(
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    k: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_neurons, d_model = gate_w.shape

    torch.manual_seed(seed)
    centroids = gate_w[torch.randperm(n_neurons)[:k]].clone()
    for _ in range(5):
        dists = torch.cdist(gate_w.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
        assignments = dists.argmin(dim=1)
        for i in range(k):
            mask = assignments == i
            if mask.any():
                centroids[i] = gate_w[mask].mean(dim=0)

    keys = torch.zeros(k, d_model, dtype=torch.float32)
    values = torch.zeros(k, d_model, dtype=torch.float32)

    for i in range(k):
        mask = assignments == i
        if mask.any():
            keys[i] = gate_w[mask].mean(dim=0)
            values[i] = down_w.t()[mask].mean(dim=0)

    return keys, values, assignments


def extract_mlp_activations(
    model,
    tokenizer,
    texts,
    layer: int,
    max_length: int = 128,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts gate pre-activation, post-SiLU gate values, and up projection values
    from a Qwen2.5 MLP layer.

    Returns:
        gate_pre: Tensor[N, 8960] — inputs to SiLU activation
        gate_post: Tensor[N, 8960] — outputs of SiLU activation
        up_out: Tensor[N, 8960] — outputs of up projection
    """
    mlp = model.model.layers[layer].mlp

    gate_data = []
    up_data = []

    def gate_hook(module, input, output):
        gate_data.append(output.detach().cpu())

    def up_hook(module, input, output):
        up_data.append(output.detach().cpu())

    handle_gate = mlp.gate_proj.register_forward_hook(gate_hook)
    handle_up = mlp.up_proj.register_forward_hook(up_hook)

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            model(**inputs)

    handle_gate.remove()
    handle_up.remove()

    if not gate_data or not up_data:
        return torch.zeros(0, 0), torch.zeros(0, 0), torch.zeros(0, 0)

    gate_pre = torch.cat([g.reshape(-1, g.shape[-1]) for g in gate_data], dim=0)
    up_out = torch.cat([u.reshape(-1, u.shape[-1]) for u in up_data], dim=0)
    gate_post = torch.nn.functional.silu(gate_pre.float())

    return gate_pre.float(), gate_post.float(), up_out.float()


def extract_mlp_activation_based(
    model,
    tokenizer,
    texts,
    layer: int,
    n_clusters: int = 256,
    method: str = "contribution",
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Build MLP key-value pairs by clustering neurons based on their real activation
    patterns, not just weight vectors.

    method='contribution': Rank neurons by effective contribution magnitude,
        bin into n_clusters groups.
    method='correlation': Compute per-neuron activation correlation, cluster
        by similar firing patterns.
    """
    mlp = model.model.layers[layer].mlp
    gate_w = mlp.gate_proj.weight.data.clone().float().cpu()
    up_w = mlp.up_proj.weight.data.clone().float().cpu()
    down_w = mlp.down_proj.weight.data.clone().float().cpu()
    n_neurons = gate_w.shape[0]

    gate_pre, gate_post, up_out = extract_mlp_activations(
        model, tokenizer, texts, layer, device=device
    )

    if gate_pre.shape[0] == 0:
        return {
            "keys": torch.zeros(1, gate_w.shape[1]),
            "values": torch.zeros(1, gate_w.shape[1]),
            "method": "activation_based_empty",
            "n_neurons": n_neurons,
            "n_keys": 1,
        }

    eff = (gate_post * up_out).abs().mean(dim=0)
    eff = eff / eff.sum()

    if method == "contribution":
        sorted_idx = torch.argsort(eff, descending=True)
        keys = torch.zeros(n_clusters, gate_w.shape[1])
        values = torch.zeros(n_clusters, gate_w.shape[1])

        per_cluster = n_neurons // n_clusters
        for k in range(n_clusters):
            start = k * per_cluster
            end = (k + 1) * per_cluster if k < n_clusters - 1 else n_neurons
            idxs = sorted_idx[start:end]
            if len(idxs) == 0:
                continue
            w = eff[idxs] / eff[idxs].sum()
            keys[k] = (gate_w[idxs] * w.unsqueeze(-1)).sum(dim=0)
            values[k] = (down_w.t()[idxs] * w.unsqueeze(-1)).sum(dim=0)

        method_used = "activation_contribution_binned"

    elif method == "correlation":
        n_samples = min(gate_pre.shape[0], 200)
        acts_sample = gate_pre[:n_samples].float()

        n_corr = min(n_neurons, 2048)
        top_neurons = torch.argsort(eff, descending=True)[:n_corr]

        corr_vecs = torch.zeros(n_corr, n_corr // 4)
        subsample = top_neurons[torch.randperm(n_corr)[: n_corr // 4]]

        for j, anchor_idx in enumerate(subsample):
            anchor = acts_sample[:, anchor_idx]
            for i, neuron_idx in enumerate(top_neurons):
                target = acts_sample[:, neuron_idx]
                an = anchor - anchor.mean()
                tn = target - target.mean()
                denom = (an.norm() * tn.norm()) + 1e-8
                corr_vecs[i, j] = (an * tn).sum() / denom

        from sklearn.cluster import KMeans
        k = min(n_clusters, n_corr)
        corr_np = corr_vecs.numpy()
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        assignments = torch.tensor(km.fit_predict(corr_np), dtype=torch.long)

        keys = torch.zeros(k, gate_w.shape[1])
        values = torch.zeros(k, gate_w.shape[1])
        for cl in range(k):
            mask = assignments == cl
            if not mask.any():
                continue
            top_idxs = top_neurons[mask]
            w = eff[top_idxs] / eff[top_idxs].sum()
            keys[cl] = (gate_w[top_idxs] * w.unsqueeze(-1)).sum(dim=0)
            values[cl] = (down_w.t()[top_idxs] * w.unsqueeze(-1)).sum(dim=0)

        n_clusters = k
        method_used = "activation_correlation_kmeans"

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "keys": keys,
        "values": values,
        "method": method_used,
        "n_neurons": n_neurons,
        "n_keys": n_clusters,
    }


def extract_gated_mlp_sparse(
    model,
    layer: int,
    n_clusters: int = 256,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    Decompose gated SiLU MLP into sparse lookup format:
    
    For each neuron cluster i:
      gate_score = SiLU(gate_key_i @ x + gate_bias_i)
      output += gate_score * value_i * scale_i
    
    This captures the gating non-linearity that simple key-value
    dot-product lookup misses.
    """
    mlp = model.model.layers[layer].mlp
    gate_w = mlp.gate_proj.weight.data.clone().float().cpu()
    gate_b = mlp.gate_proj.bias
    if gate_b is not None:
        gate_b = gate_b.data.clone().float().cpu()
    else:
        gate_b = torch.zeros(gate_w.shape[0])
    up_w = mlp.up_proj.weight.data.clone().float().cpu()
    down_w = mlp.down_proj.weight.data.clone().float().cpu()
    n_neurons = gate_w.shape[0]
    d_model = gate_w.shape[1]

    k = max(2, min(n_clusters, n_neurons // 8))

    torch.manual_seed(seed)
    centroids = gate_w[torch.randperm(n_neurons)[:k]].clone()
    for _ in range(5):
        dists = torch.cdist(gate_w.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
        assignments = dists.argmin(dim=1)
        for i in range(k):
            mask = assignments == i
            if mask.any():
                centroids[i] = gate_w[mask].mean(dim=0)

    gate_keys = torch.zeros(k, d_model, dtype=torch.float32)
    gate_biases = torch.zeros(k, dtype=torch.float32)
    values = torch.zeros(k, d_model, dtype=torch.float32)
    scales = torch.zeros(k, dtype=torch.float32)

    for i in range(k):
        mask = assignments == i
        if mask.any():
            n_in_cluster = mask.sum().float()
            gate_keys[i] = gate_w[mask].mean(dim=0)
            gate_biases[i] = gate_b[mask].mean()
            values[i] = down_w.t()[mask].mean(dim=0)
            scales[i] = n_in_cluster / n_neurons

    values_norm = values.norm(dim=1, keepdim=True) + 1e-8
    values = values / values_norm

def collect_mlp_io_pairs(
    model,
    tokenizer,
    layer: int,
    prompts: list,
    max_length: int = 96,
    device: str = "cuda",
) -> tuple:
    """
    Collect (input, output) pairs from a specific MLP layer
    for distillation training. Returns (inputs, targets) tensors.
    """
    mlp = model.model.layers[layer].mlp
    mlp_inputs = []
    mlp_outputs = []

    def pre_hook(module, args, kwargs):
        if args:
            mlp_inputs.append(args[0].detach().clone())
        elif "hidden_states" in kwargs:
            mlp_inputs.append(kwargs["hidden_states"].detach().clone())

    def post_hook(module, args, output):
        mlp_outputs.append((output[0] if isinstance(output, tuple) else output).detach().clone())

    ph = mlp.register_forward_pre_hook(pre_hook, with_kwargs=True)
    pth = mlp.register_forward_hook(post_hook)

    for prompt in prompts:
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            model(**inp)

    pth.remove()
    ph.remove()

    inputs = torch.cat([t.reshape(-1, t.shape[-1]) for t in mlp_inputs], dim=0).float()
    targets = torch.cat([t.reshape(-1, t.shape[-1]) for t in mlp_outputs], dim=0).float()

    return inputs, targets


def extract_full_mlp_weights(
    model,
    layer: int,
) -> Dict[str, torch.Tensor]:
    """
    Extract raw MLP weights for sparse_down_projection operator.
    No clustering or decomposition — these are the exact weights.

    Returns:
        gate_weight: Tensor[8960, 1536]
        gate_bias: Tensor[8960] or None
        up_weight: Tensor[8960, 1536]  
        down_weight: Tensor[1536, 8960]
        n_neurons: int
    """
    mlp = model.model.layers[layer].mlp

    gate_w = mlp.gate_proj.weight.data.clone().float().cpu()
    gate_b = mlp.gate_proj.bias
    if gate_b is not None:
        gate_b = gate_b.data.clone().float().cpu()
    up_w = mlp.up_proj.weight.data.clone().float().cpu()
    down_w = mlp.down_proj.weight.data.clone().float().cpu()

    return {
        "gate_weight": gate_w,
        "gate_bias": gate_b,
        "up_weight": up_w,
        "down_weight": down_w,
        "n_neurons": gate_w.shape[0],
    }


def extract_full_mlp_weights_lr(
    model,
    layer: int,
    rank: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Extract MLP weights with low-rank SVD of the down projection matrix.
    Used by sparse_down_projection_lr operator for residual correction.

    Returns same as extract_full_mlp_weights plus:
        U_r: Tensor[1536, R] — low-rank output basis
        V_r: Tensor[R, 8960] — low-rank coefficient mapping
    """
    weights = extract_full_mlp_weights(model, layer)
    down_w = weights["down_weight"]
    U, S, Vh = torch.linalg.svd(down_w, full_matrices=False)
    r = min(rank, len(S))
    U_r = U[:, :r] @ torch.diag(S[:r])
    V_r = Vh[:r, :]
    weights["U_r"] = U_r
    weights["V_r"] = V_r
    weights["svd_rank"] = r
    return weights
