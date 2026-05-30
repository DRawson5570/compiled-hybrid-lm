from __future__ import annotations

import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from sae_editor.tests.utils import (
    SyntheticConfig,
    SyntheticModel,
    make_random_sae,
)


@pytest.fixture(scope="session")
def tiny_gpt2_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def tiny_gpt2_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "sshleifer/tiny-gpt2",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    return model


@pytest.fixture
def synthetic_model():
    config = SyntheticConfig(d_model=64, n_layers=3, n_heads=4, vocab_size=1000)
    return SyntheticModel(config)


@pytest.fixture
def synthetic_sae_factory():
    def _make(d_model=64, n_features=32, l1_lambda=1e-3):
        return make_random_sae(d_model, n_features, l1_lambda)
    return _make


@pytest.fixture
def synthetic_sae(synthetic_sae_factory):
    return synthetic_sae_factory()


@pytest.fixture
def temp_safetensors():
    tensors = {
        "model.layers.0.mlp.down_proj.weight": torch.randn(8, 4),
        "model.layers.0.mlp.up_proj.weight": torch.randn(4, 8),
        "model.layers.1.mlp.down_proj.weight": torch.randn(8, 4),
        "model.layers.1.mlp.up_proj.weight": torch.randn(4, 8),
        "model.embed_tokens.weight": torch.randn(1000, 8),
    }
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    save_file(tensors, path)
    yield path
    os.unlink(path)
