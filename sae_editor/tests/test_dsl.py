"""Wave 5: NRTCS DSL parser tests."""

from __future__ import annotations

import pytest
import torch

from sae_editor.dsl.nrtcs_parser import parse_nrtcs, serialize_nrtcs, ParseError


class TestParseNRTCS:
    def test_parse_single_layer_dense_map(self):
        source = """
        layer 14 {
            override mlp.down_proj = dense_map(
                < 0.9, -0.1, 0.1, 0.1 >,
                < 0.1, 0.1, 0.9, 0.1 >
            );
        }
        """
        result = parse_nrtcs(source)
        assert 14 in result
        assert result[14]["keys"].shape == (1, 4)
        assert result[14]["values"].shape == (1, 4)

    def test_parse_multiple_layers(self):
        source = """
        layer 0 {
            override x = dense_map(< 1.0, 0.0 >, < 0.0, 1.0 >);
        }
        layer 5 {
            override y = dense_map(< 2.0, 1.0 >, < 3.0, 4.0 >);
        }
        """
        result = parse_nrtcs(source)
        assert set(result.keys()) == {0, 5}

    def test_parse_multiple_pairs(self):
        source = """
        layer 1 {
            override x = dense_map(
                < 1.0, 0.0 >, < 0.0, 1.0 >,
                < 2.0, 3.0 >, < 4.0, 5.0 >
            );
        }
        """
        result = parse_nrtcs(source)
        assert result[1]["keys"].shape == (2, 2)
        assert result[1]["values"].shape == (2, 2)

    def test_parse_comments_ignored(self):
        source = """
        # This is a comment
        layer 3 {  # inline comment
            override x = dense_map(< 1.0 >, < 2.0 >);
        }
        """
        result = parse_nrtcs(source)
        assert 3 in result

    def test_parse_bracket_syntax(self):
        source = """
        layer 4 {
            override x = dense_map(
                [ 0.5, -0.3 ], [ 0.7, 0.2 ]
            );
        }
        """
        result = parse_nrtcs(source)
        assert result[4]["keys"].shape == (1, 2)

    def test_parse_empty_program(self):
        assert parse_nrtcs("") == {}

    def test_parse_accepts_missing_semicolon(self):
        source = """
        layer 1 {
            override x = dense_map(< 1.0 >, < 2.0 >)
        }
        """
        result = parse_nrtcs(source)
        assert 1 in result


class TestSerializeNRTCS:
    def test_serialize_round_trip(self):
        original = {
            14: {
                "keys": torch.tensor([[0.9, -0.1, 0.1, 0.1]]),
                "values": torch.tensor([[0.1, 0.1, 0.9, 0.1]]),
            }
        }
        text = serialize_nrtcs(original)
        parsed = parse_nrtcs(text)
        assert 14 in parsed
        assert torch.allclose(parsed[14]["keys"], original[14]["keys"], atol=1e-4)

    def test_serialize_multilayer_round_trip(self):
        original = {
            0: {
                "keys": torch.tensor([[1.0, 0.0]]),
                "values": torch.tensor([[0.0, 1.0]]),
            },
            2: {
                "keys": torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
                "values": torch.tensor([[6.0, 7.0], [8.0, 9.0]]),
            },
        }
        text = serialize_nrtcs(original)
        parsed = parse_nrtcs(text)
        assert set(parsed.keys()) == {0, 2}
        assert torch.allclose(parsed[2]["keys"], original[2]["keys"], atol=1e-4)
