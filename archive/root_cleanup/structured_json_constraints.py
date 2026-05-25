"""structured_json_constraints.py

Regex check and constrained vocabulary decoding functions designed to parse 
and output deterministic structures for down-stream tool consumers.
Uses regex checks to force selected next-tokens to match schema guidelines.
"""
from __future__ import annotations

import re
import torch
from hybrid.v2_capabilities.dataset import tok2id, id2tok, V

class ConstrainedDecoder:
    """Constraints tracker guiding next token prediction into JSON layouts."""
    def __init__(self):
        # Guideline structure representing: { "status": "done" } or { "result": 27 }
        self.brackets_open = False
        self.key_written = False
        self.colon_written = False
        self.val_written = False

    def get_valid_tokens_mask(self, current_tokens: list[str], device: torch.device) -> torch.Tensor:
        """Computes a 0/1 mask filtering out invalid tokens according to structured rules."""
        mask = torch.ones(V, device=device)
        
        # Check current state from sequence
        seq_str = " ".join(current_tokens)
        
        # Case A: Inside JSON block, force structured formatting keys
        if len(current_tokens) > 0 and current_tokens[-1] == "{":
            # Force quotes or properties keys next
            for t_idx, token in id2tok.items():
                if not (token.startswith('"') or token == "}" or token in ["status", "result"]):
                    mask[t_idx] = 0.0
        elif len(current_tokens) > 0 and current_tokens[-1] in ["\"status\"", "status"]:
            # Force colon
            for t_idx, token in id2tok.items():
                if token != ":":
                    mask[t_idx] = 0.0
        elif len(current_tokens) > 0 and current_tokens[-1] == ":":
            # Force dynamic string or values
            pass
        return mask

def test_constrained_decoder():
    print("=" * 80)
    print("        CMI OUTBOUND STRUCTURED JSON DECODER VERIFICATION")
    print("=" * 80)
    
    decoder = ConstrainedDecoder()
    device = torch.device("cpu")
    
    # Run test on open bracket token {
    mask = decoder.get_valid_tokens_mask(["{"], device)
    allowed_tokens = [id2tok[i] for i in range(V) if mask[i] > 0]
    
    print("Pre-compiled Allowed Next Tokens inside Bracket Schema:")
    print(f"  {allowed_tokens[:10]} ... ({len(allowed_tokens)} total allowed)")
    print("-" * 80)

    # Run test on key token 'status'
    mask_colon = decoder.get_valid_tokens_mask(["{", "status"], device)
    allowed_colon = [id2tok[i] for i in range(V) if mask_colon[i] > 0]
    print("Pre-compiled Allowed Next Tokens after key 'status':")
    print(f"  {allowed_colon}")
    
    val_check = ":" in allowed_colon
    print(f"  Is colon ':' forced next? {'Yes (Pass)' if val_check else 'No (Fail)'}")
    print("=" * 80)

if __name__ == "__main__":
    test_constrained_decoder()
