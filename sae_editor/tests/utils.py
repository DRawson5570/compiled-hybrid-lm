from __future__ import annotations

import torch
import torch.nn as nn


class SyntheticConfig:
    def __init__(self, d_model=64, n_layers=3, n_heads=4, vocab_size=1000):
        self.hidden_size = d_model
        self.num_hidden_layers = n_layers
        self.num_attention_heads = n_heads
        self.num_key_value_heads = n_heads
        self.vocab_size = vocab_size


class SyntheticLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return (x + self.linear(x),)


class SyntheticModel(nn.Module):
    def __init__(self, config: SyntheticConfig):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [SyntheticLayer(config.hidden_size) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, input_ids=None, **kwargs):
        if input_ids is not None:
            B, T = input_ids.shape
        else:
            B, T = 1, 8
            input_ids = torch.zeros(B, T, dtype=torch.long)
        device = input_ids.device if isinstance(input_ids, torch.Tensor) else torch.device("cpu")

        generator = torch.Generator(device=device)
        generator.manual_seed(12345)
        x = torch.randn(
            B, T, self.config.hidden_size,
            generator=generator, device=device,
        )
        for i, layer in enumerate(self.model.layers):
            x_raw = layer(x + 0.001 * i)
            x = x_raw[0] if isinstance(x_raw, tuple) else x_raw
        return type("Output", (), {
            "logits": torch.randn(B, T, self.config.vocab_size, generator=generator, device=device),
        })()


def make_random_sae(
    d_model: int = 64,
    n_features: int = 32,
    l1_lambda: float = 1e-3,
):
    from full_compiled_experiment.ucn.decompile.sae import SparseAutoencoder

    sae = SparseAutoencoder(
        d_model=d_model,
        n_features=n_features,
        l1_lambda=l1_lambda,
        tied_decoder=False,
    )
    sae.eval()
    return sae


def assert_tensors_close(a, b, atol=1e-4, msg=""):
    if not torch.allclose(a, b, atol=atol):
        max_diff = (a.float() - b.float()).abs().max().item()
        raise AssertionError(
            f"{msg}Max diff: {max_diff:.6f} (atol={atol})"
        )
