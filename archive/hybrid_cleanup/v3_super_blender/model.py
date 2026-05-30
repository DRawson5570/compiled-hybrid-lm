"""hybrid/v3_super_blender/model.py

Highly advanced, sequence-aware routing blenders (GRU, Causal CNN, and Lookback MLP)
for combining compiled wikitext channels.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUBlender(nn.Module):
    """Multi-layer unidirectional GRU sequence router.
    
    This model processes the continuous token stream causally.
    """
    def __init__(self, in_dim: int, n_channels: int, hidden: int = 256,
                 num_layers: int = 2, dropout: float = 0.1, init_uniform: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.n_channels = n_channels
        self.hidden = hidden
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_channels)
        )
        if init_uniform:
            # Shift predictions towards uniform initialization if requested
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape (B, SeqLen, in_dim) or (SeqLen, in_dim)
            h0: optional shape (num_layers, B, hidden)
        Returns:
            log_w: log-probabilities of shape (B, SeqLen, n_channels)
            h_n: final hidden state of shape (num_layers, B, hidden)
        """
        is_flat = (x.dim() == 2)

        if is_flat:
            x = x.unsqueeze(0)  # Add batch dim -> (1, T, in_dim)
            
        out, h_n = self.gru(x, h0)
        logits = self.classifier(out)
        log_w = F.log_softmax(logits, dim=-1)
        
        if is_flat:
            log_w = log_w.squeeze(0)  # -> (T, n_channels)
            
        return log_w, h_n


class WindowMLPBlender(nn.Module):
    """Functionally identical to TinyBlender but expanded with a lookback window W."""
    def __init__(self, single_step_dim: int, n_channels: int, lookback_window: int = 16,
                 hidden: int = 128, dropout: float = 0.0, init_uniform: bool = True):
        super().__init__()
        self.single_step_dim = single_step_dim
        self.n_channels = n_channels
        self.lookback_window = lookback_window
        in_dim = single_step_dim * lookback_window
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_channels),
        )
        if init_uniform:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def build_windowed_features(self, features: torch.Tensor) -> torch.Tensor:
        T, F_dim = features.shape
        W = self.lookback_window
        padded = F.pad(features, (0, 0, W - 1, 0), mode="constant", value=0.0)
        if W > 1:
            padded[:W - 1] = features[0]
        indices = torch.arange(T, device=features.device).unsqueeze(1) + torch.arange(W, device=features.device)
        windowed = padded[indices]  # (T, W, F_dim)
        return windowed.reshape(T, -1)

    def forward(self, features: torch.Tensor, is_already_windowed: bool = False) -> torch.Tensor:
        if not is_already_windowed:
            features = self.build_windowed_features(features)
        logits = self.net(features)
        return F.log_softmax(logits, dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LookbackMLPBlender(nn.Module):
    """Lookback MLP Blender.
    
    Concatenates features from the past W steps and processes them with a ResNet MLP.
    This architecture is extremely fast, highly parallelizable, has zero state-drift,
    and is extremely robust.
    """
    def __init__(self, single_step_dim: int, n_channels: int, lookback_window: int = 16,
                 hidden: int = 256, num_layers: int = 2, dropout: float = 0.1, init_uniform: bool = True):
        super().__init__()
        self.single_step_dim = single_step_dim
        self.n_channels = n_channels
        self.lookback_window = lookback_window
        
        in_dim = single_step_dim * lookback_window
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.res_layers = nn.ModuleList([
            ResidualBlock(hidden, dropout=dropout) for _ in range(num_layers)
        ])
        
        self.out_proj = nn.Linear(hidden, n_channels)
        
        if init_uniform:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def build_windowed_features(self, features: torch.Tensor) -> torch.Tensor:
        """Constructs windowed features from single-step features (T, F_dim).
        
        For position t, concatenates features[t - lookback_window + 1 : t + 1].
        Pads t < 0 with features[0] (constant padding).
        """
        T, F_dim = features.shape
        W = self.lookback_window
        
        # We can construct the sliding lookback window efficiently using unfolding or indexing
        # Pad features on the left for lookback
        padded = F.pad(features, (0, 0, W - 1, 0), mode="constant", value=0.0)
        # Re-fill the padded initial steps with features[0] (constant replication)
        if W > 1:
            padded[:W - 1] = features[0]
            
        # Unfold/stride along dim 0
        # padded has length T + W - 1. We want to extract slices of size W starting at 0, 1, ..., T-1
        # Strided view into (T, W, F_dim)
        # stride_0, stride_1 = padded.stride()
        # windowed = padded.as_strided((T, W, F_dim), (stride_0, stride_0, stride_1))
        # Alternatively, a simple loop or indexing to be 100% robust:
        # Since T is typically ~30k to 100k, we can use indexing for convenience and absolute safety:
        indices = torch.arange(T, device=features.device).unsqueeze(1) + torch.arange(W, device=features.device)
        windowed = padded[indices]  # (T, W, F_dim)
        
        # Flatten the window dimension: (T, W * F_dim)
        return windowed.reshape(T, -1)

    def forward(self, features: torch.Tensor, is_already_windowed: bool = False) -> torch.Tensor:
        """
        Args:
            features: if is_already_windowed is False: shape (T, single_step_dim)
                      if is_already_windowed is True: shape (T, single_step_dim * lookback_window)
        """
        if not is_already_windowed:
            features = self.build_windowed_features(features)
            
        x = self.in_proj(features)
        for layer in self.res_layers:
            x = layer(x)
        logits = self.out_proj(x)
        return F.log_softmax(logits, dim=-1)


class CausalConvBlender(nn.Module):
    """1D Causal Convolutional Network sequence router.
    
    Uses standard dilated convolutions with casual masking so it remains 100% causal.
    """
    def __init__(self, in_dim: int, n_channels: int, channels: int = 128,
                 kernel_size: int = 3, num_layers: int = 3, dropout: float = 0.1, init_uniform: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.n_channels = n_channels
        self.channels = channels
        
        # Initial proj
        self.in_proj = nn.Conv1d(in_dim, channels, kernel_size=1)
        
        # Dilated causal blocks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation  # Padding size to ensure causality
            self.layers.append(
                nn.ModuleDict({
                    "conv": nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                      padding=padding, dilation=dilation),
                    "norm": nn.LayerNorm(channels),
                    "dropout": nn.Dropout(dropout)
                })
            )
            
        self.out_proj = nn.Conv1d(channels, n_channels, kernel_size=1)
        self.kernel_size = kernel_size
        
        if init_uniform:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (B, SeqLen, in_dim) or (SeqLen, in_dim)
        """
        is_flat = (x.dim() == 2)
        if is_flat:
            x = x.unsqueeze(0)  # (1, T, in_dim)
            
        # Conv1d expects (B, in_dim, SeqLen)
        h = x.transpose(1, 2)
        h = self.in_proj(h)
        
        for layer in self.layers:
            residual = h
            # Apply dilated conv with padding
            out = layer["conv"](h)
            # Chop off the end to keep causal (padding is on left & right, we only want left padding)
            # With padding = (kernel_size-1)*dilation, conv output length is SeqLen + padding.
            # We discard the last `padding` elements.
            padding_len = out.shape[-1] - h.shape[-1]
            if padding_len > 0:
                out = out[..., :-padding_len]
                
            # layer norm expects (B, SeqLen, channels) -> transpose and transpose back
            out_norm = layer["norm"](out.transpose(1, 2)).transpose(1, 2)
            out_act = F.gelu(out_norm)
            h = residual + layer["dropout"](out_act)
            
        logits = self.out_proj(h)  # (B, n_channels, SeqLen)
        logits = logits.transpose(1, 2)  # (B, SeqLen, n_channels)
        log_w = F.log_softmax(logits, dim=-1)
        
        if is_flat:
            log_w = log_w.squeeze(0)  # -> (T, n_channels)
            
        return log_w
