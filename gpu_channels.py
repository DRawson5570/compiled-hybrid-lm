"""gpu_channels.py — GPU-vectorized compiled channel features with KV, Topic, POS.
Zero nested Python loops — pure vectorized tensor ops to prevent CUDA sync stalls.
"""
import torch


@torch.no_grad()
def compute_gpu_kv_cache(input_ids: torch.Tensor, ppmi_embeddings: torch.Tensor,
                         window: int = 128):
    B, T = input_ids.shape; device = input_ids.device
    emb = ppmi_embeddings[input_ids.clamp(0, ppmi_embeddings.shape[0] - 1)]
    scores = torch.matmul(emb, emb.transpose(1, 2))
    norms = emb.norm(p=2, dim=-1, keepdim=True)
    norm_matrix = torch.matmul(norms, norms.transpose(1, 2)).clamp(min=1e-8)
    cos_sim = scores / norm_matrix
    causal_mask = torch.tril(torch.ones(T, T, device=device), diagonal=-1)
    window_mask = torch.triu(torch.ones(T, T, device=device), diagonal=-window)
    masked_sim = cos_sim * causal_mask * window_mask
    max_retrieval, _ = masked_sim.max(dim=-1)
    return max_retrieval


@torch.no_grad()
def compute_gpu_topic_prior(input_ids: torch.Tensor, topic_matrix: torch.Tensor,
                            decay: float = 0.95):
    B, T = input_ids.shape; device = input_ids.device
    topic_embs = topic_matrix[input_ids.clamp(0, topic_matrix.shape[0] - 1)]
    idx = torch.arange(T, device=device)
    weight_matrix = torch.pow(decay, (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0).float())
    weight_matrix = torch.tril(weight_matrix) * (1.0 - decay)
    weight_matrix[:, 0] = torch.pow(decay, idx.float())
    running_topics = torch.matmul(topic_embs.transpose(1, 2), weight_matrix.T).transpose(1, 2)
    return (running_topics * topic_embs).sum(dim=-1)


class GPUFeatureComputer:
    """Computes 21 channel features on GPU with zero Python loop stalls."""
    
    def __init__(self, V=50257, punct_ids=None, topic_matrix=None,
                 pos_tags=None, ppmi_embeddings=None, device='cuda'):
        self.V = V; self.device = device
        self.punct_ids = torch.tensor(list(punct_ids or []), device=device)
        self.topic_matrix = topic_matrix.to(device) if topic_matrix is not None else None
        self.ppmi_embeddings = ppmi_embeddings.to(device) if ppmi_embeddings is not None else None
        if pos_tags is not None:
            self.pos_tags = torch.tensor([pos_tags.get(i, 0) for i in range(V)], dtype=torch.long, device=device)
        else:
            self.pos_tags = torch.zeros(V, dtype=torch.long, device=device)
    
    def compute_features(self, input_ids):
        B, T = input_ids.shape; device = self.device; feats = torch.zeros(B, T, 21, device=device)
        
        # Channels 0-8: overwritten by CPU loader, leave as zeros
        
        # 15: Punct density
        punct_mask = torch.isin(input_ids, self.punct_ids).float()
        feats[:, :, 15] = punct_mask.cumsum(dim=1) / torch.arange(1, T+1, device=device).float().unsqueeze(0)
        
        # 16: Repetition (adjacent token match)
        prev = torch.roll(input_ids, 1, dims=1); prev[:, 0] = -1
        feats[:, :, 16] = (input_ids == prev).float()
        
        # 17: Unique ratio (1-step + 2-step repeats)
        prev2 = torch.roll(input_ids, 2, dims=1); prev2[:, :2] = -2
        reps = (input_ids == prev).float() + (input_ids == prev2).float()
        feats[:, :, 17] = 1.0 - (reps.clamp(max=1.0) * 0.5)
        
        # 18: Topic prior
        if self.topic_matrix is not None and T >= 2:
            feats[:, :, 18] = compute_gpu_topic_prior(input_ids, self.topic_matrix, decay=0.95)
        
        # 19: KV retrieval
        if self.ppmi_embeddings is not None and T >= 2:
            feats[:, :, 19] = compute_gpu_kv_cache(input_ids, self.ppmi_embeddings, window=64)
        
        # 20: POS match
        pos_ids = self.pos_tags[input_ids.clamp(0, self.V - 1)]
        prev_pos = torch.roll(pos_ids, 1, dims=1); prev_pos[:, 0] = -1
        feats[:, :, 20] = (pos_ids == prev_pos).float()
        
        return feats
    
    def reset(self):
        pass
