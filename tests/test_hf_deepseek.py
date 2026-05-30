from __future__ import annotations

import torch

from hybrid.cartridges import CartridgeManifest, CartridgeRole, SteererCartridgeRack
from hybrid.hf_deepseek import DeepSeekConfig, DeepSeekForCausalLM


class AdditiveSteerer(torch.nn.Module):
    def set_weights(self, weights):
        self.weights = weights

    def _steer_layer(self, hidden, layer_idx):
        return hidden + 0.125 * (layer_idx + 1)


def test_deepseek_backbone_exposes_cartridge_hook_surface():
    config = DeepSeekConfig(vocab_size=32, d_model=12, n_layers=2, n_heads=3, d_ff=24, max_len=16)
    model = DeepSeekForCausalLM(config)
    rack = SteererCartridgeRack()
    manifest = CartridgeManifest(
        cartridge_id='test-steerer',
        role=CartridgeRole.SUPERPOSITION_STEERER,
        base_model_id='deepseek-test',
        tokenizer_id='gpt2-bpe',
        channel_schema='cmi-21ch-v3',
        inject_layers=(0, 1),
    )
    rack.mount(manifest, AdditiveSteerer())

    assert rack.register_hooks(model) == 2
    output = model(torch.randint(0, config.vocab_size, (2, 5)))

    assert output.logits.shape == (2, 5, config.vocab_size)