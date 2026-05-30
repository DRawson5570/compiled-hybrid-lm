"""Test binary splicer: safetensors file I/O and inline patching."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
from safetensors.torch import save_file

from sae_editor.splicer import SafetensorsSplicer, splice_tensor


class TestSafetensorsSplicer:
    @pytest.fixture
    def temp_safetensors(self):
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

    def test_open_and_parse_header(self, temp_safetensors):
        with SafetensorsSplicer(temp_safetensors) as spl:
            assert len(spl.tensor_names) == 5
            assert "model.layers.0.mlp.down_proj.weight" in spl.tensor_names

    def test_get_tensor_info(self, temp_safetensors):
        with SafetensorsSplicer(temp_safetensors) as spl:
            shape = spl.get_tensor_shape("model.layers.0.mlp.down_proj.weight")
            assert shape == [8, 4]

    def test_read_tensor(self, temp_safetensors):
        with SafetensorsSplicer(temp_safetensors) as spl:
            data = spl.read_tensor("model.embed_tokens.weight")
            assert len(data) > 0

    def test_splice_tensor_replace(self, temp_safetensors):
        new_down = torch.zeros(8, 4)
        new_bytes = new_down.numpy().tobytes()

        with SafetensorsSplicer(temp_safetensors) as spl:
            spl.splice_tensor("model.layers.0.mlp.down_proj.weight", new_bytes)

        from safetensors import safe_open
        with safe_open(temp_safetensors, framework="pt") as f:
            loaded = f.get_tensor("model.layers.0.mlp.down_proj.weight")

        assert torch.allclose(loaded, new_down)

    def test_splice_tensor_size_mismatch(self, temp_safetensors):
        wrong_size = torch.zeros(16, 4).numpy().tobytes()
        with SafetensorsSplicer(temp_safetensors) as spl:
            with pytest.raises(AssertionError, match="Size mismatch"):
                spl.splice_tensor("model.layers.0.mlp.down_proj.weight", wrong_size)

    def test_splice_tensor_no_verify(self, temp_safetensors):
        wider = torch.zeros(8, 8).numpy().tobytes()
        with SafetensorsSplicer(temp_safetensors) as spl:
            with pytest.raises(IndexError):
                spl.splice_tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    wider,
                    verify_shape=False,
                )

    def test_splice_mlp_convenience(self, temp_safetensors):
        new_down = torch.ones(8, 4) * 0.5
        new_up = torch.ones(4, 8) * -1.0

        with SafetensorsSplicer(temp_safetensors) as spl:
            spl.splice_mlp(layer=0, W_down=new_down, W_up=new_up)

        from safetensors import safe_open
        with safe_open(temp_safetensors, framework="pt") as f:
            loaded_down = f.get_tensor("model.layers.0.mlp.down_proj.weight")
            loaded_up = f.get_tensor("model.layers.0.mlp.up_proj.weight")

        assert torch.allclose(loaded_down, new_down)
        assert torch.allclose(loaded_up, new_up)

    def test_other_layers_untouched(self, temp_safetensors):
        from safetensors import safe_open
        with safe_open(temp_safetensors, framework="pt") as f:
            original_l1 = f.get_tensor("model.layers.1.mlp.down_proj.weight").clone()

        new_down = torch.zeros(8, 4)
        new_up = torch.zeros(4, 8)
        with SafetensorsSplicer(temp_safetensors) as spl:
            spl.splice_mlp(layer=0, W_down=new_down, W_up=new_up)

        with safe_open(temp_safetensors, framework="pt") as f:
            loaded_l1 = f.get_tensor("model.layers.1.mlp.down_proj.weight")

        assert torch.allclose(loaded_l1, original_l1)

    def test_splice_tensor_convenience_function(self, temp_safetensors):
        new_embed = torch.ones(1000, 8)
        new_bytes = new_embed.numpy().tobytes()
        splice_tensor(temp_safetensors, "model.embed_tokens.weight", new_bytes)

        from safetensors import safe_open
        with safe_open(temp_safetensors, framework="pt") as f:
            loaded = f.get_tensor("model.embed_tokens.weight")
        assert torch.allclose(loaded, new_embed)

    def test_keyerror_nonexistent_tensor(self, temp_safetensors):
        with SafetensorsSplicer(temp_safetensors) as spl:
            with pytest.raises(KeyError):
                spl.splice_tensor("nonexistent.tensor", b"\x00" * 32)
