from __future__ import annotations

import re

import torch


class ParseError(ValueError):
    pass


def parse_nrtcs(source: str) -> dict[int, dict[str, torch.Tensor]]:
    """Parse NRTCS DSL text into edit dict.

    Grammar (from NRTCS_SPEC.md §3.1):
        layer N { override ID = dense_map(<vec, vec>, ...); }

    Returns {layer: {"keys": (N, d_in), "values": (N, d_out)}}
    """
    source = _strip_comments(source)
    result = {}
    pos = 0

    while pos < len(source):
        pos = _skip_whitespace(source, pos)
        if pos >= len(source):
            break

        pos, layer_idx = _parse_layer_header(source, pos)
        pos = _skip_whitespace(source, pos)
        if pos >= len(source) or source[pos] != "{":
            raise ParseError(f"Expected '{{' after layer {layer_idx} at position {pos}")
        pos += 1

        pairs = []
        while pos < len(source):
            pos = _skip_whitespace(source, pos)
            if pos >= len(source):
                raise ParseError(f"Unclosed layer {layer_idx} block")
            if source[pos] == "}":
                pos += 1
                break

            pos = _parse_override_stmt(source, pos)
            pos, kv_pairs = _parse_dense_map(source, pos)
            pairs.extend(kv_pairs)
            pos = _skip_whitespace(source, pos)
            if pos < len(source) and source[pos] == ";":
                pos += 1

        if pairs:
            keys_list = [k for k, v in pairs]
            values_list = [v for k, v in pairs]
            result[layer_idx] = {
                "keys": torch.stack(keys_list, dim=0),
                "values": torch.stack(values_list, dim=0),
            }

    return result


def serialize_nrtcs(edits: dict[int, dict[str, torch.Tensor]]) -> str:
    lines = []
    for layer_idx, edit in sorted(edits.items()):
        keys = edit["keys"]
        values = edit["values"]
        lines.append(f"layer {layer_idx} {{")
        for i in range(keys.shape[0]):
            key_vec = _vec_to_str(keys[i])
            val_vec = _vec_to_str(values[i])
            lines.append(f"    override PRM_{i} = dense_map(")
            lines.append(f"        < {key_vec} >, < {val_vec} >")
            lines.append("    );")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def _strip_comments(source: str) -> str:
    return re.sub(r"#.*$", "", source, flags=re.MULTILINE)


def _skip_whitespace(s: str, pos: int) -> int:
    while pos < len(s) and s[pos] in " \t\n\r":
        pos += 1
    return pos


def _parse_layer_header(s: str, pos: int) -> tuple[int, int]:
    if not s[pos:pos+5].lower() == "layer":
        raise ParseError(f"Expected 'layer' at position {pos}")
    pos += 5
    pos = _skip_whitespace(s, pos)
    start = pos
    while pos < len(s) and s[pos].isdigit():
        pos += 1
    if pos == start:
        raise ParseError(f"Expected layer number at position {pos}")
    layer_idx = int(s[start:pos])
    return pos, layer_idx


def _parse_override_stmt(s: str, pos: int) -> int:
    if s[pos:pos+8].lower() != "override":
        return pos
    while pos < len(s) and s[pos] not in "={;":
        pos += 1
    if pos < len(s) and s[pos] == "=":
        pos += 1
    return pos


def _parse_dense_map(s: str, pos: int) -> tuple[int, list[tuple[torch.Tensor, torch.Tensor]]]:
    pos = _skip_whitespace(s, pos)
    tag_end = min(pos + 9, len(s))
    tag = s[pos:tag_end]
    if not tag.startswith("dense_map"):
        return pos, []

    pos += 9
    while pos < len(s) and s[pos] != "(":
        pos += 1
    if pos < len(s) and s[pos] == "(":
        pos += 1
    pairs = []
    while pos < len(s):
        pos = _skip_whitespace(s, pos)
        if pos >= len(s):
            break
        if s[pos] == ")":
            pos += 1
            break
        if s[pos] == ";":
            break
        pos, key_vec = _parse_vector(s, pos)
        pos = _skip_whitespace(s, pos)
        if pos < len(s) and s[pos] == ",":
            pos += 1
        pos = _skip_whitespace(s, pos)
        pos, val_vec = _parse_vector(s, pos)
        pairs.append((key_vec, val_vec))
        pos = _skip_whitespace(s, pos)
        if pos < len(s) and s[pos] == ",":
            pos += 1
    return pos, pairs


def _parse_vector(s: str, pos: int) -> tuple[int, torch.Tensor]:
    pos = _skip_whitespace(s, pos)
    if pos >= len(s):
        raise ParseError("Unexpected EOF while parsing vector")

    if s[pos] == "<":
        pos += 1
    elif s[pos] == "[":
        pos += 1
    else:
        raise ParseError(f"Expected '<' or '[' at position {pos}, got '{s[pos]}'")

    values = []
    while pos < len(s):
        pos = _skip_whitespace(s, pos)
        if pos >= len(s):
            break
        if s[pos] in ">]":
            pos += 1
            break
        if s[pos] == ",":
            pos += 1
            continue
        start = pos
        while pos < len(s) and s[pos] not in ",> \t\n\r]":
            pos += 1
        token = s[start:pos]
        if token:
            values.append(float(token))

    return pos, torch.tensor(values)


def _vec_to_str(vec: torch.Tensor) -> str:
    parts = [f"{float(v):.4f}" for v in vec]
    return ", ".join(parts)
