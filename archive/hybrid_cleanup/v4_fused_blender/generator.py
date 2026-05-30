"""hybrid/v4_fused_blender/generator.py

Generates fused tokens streams mixing WikiText-103 content and multi-task capability 
prompts (translations, reasoning, code, tool addition) for robust Phase II real-world training.
"""
from __future__ import annotations

import random
import torch
from pathlib import Path

def interleave_capabilities_with_wikitext(wikitext_tokens: list[str], tok2id: dict[str, int]) -> list[str]:
    """Interleaves synthetic capability tasks cleanly inside real wikitext tokens."""
    fused_tokens = []
    
    # Simple templates for interleaving
    translations = [
        ("translate dog to french", "chien"),
        ("translate cat to french", "chat"),
        ("translate gravity to french", "gravité"),
        ("translate apple to french", "pomme"),
    ]
    
    # Multi-step reasoning entities template
    def gen_reasoning():
        # E0001-E0009 are valid IDs in VOCAB_WORDS from v2_capabilities/dataset.py
        idx = random.choice([1, 4, 7])
        objs = [f"E000{idx}", f"E000{idx+1}", f"E000{idx+2}"]
        return (
            f"{objs[0]} is larger than {objs[1]} . {objs[1]} is larger than {objs[2]} . Therefore , {objs[0]} is larger than",
            objs[2]
        )
        
    # Python code templates (strictly using tokens in VOCAB_WORDS)
    code_templates = [
        ("def get_sum ( a , b )", ":"),
        ("import numpy as np", "."),
    ]
    
    # Math tool templates (strictly using expressions in VOCAB_WORDS)
    def gen_math():
        queries = [
            ("What is 54 + 23 ?", "[USE_TOOL: calculator expr= 54+23 ] Answer is 77"),
            ("What is 12 + 15 ?", "[USE_TOOL: calculator expr= 12+15 ] Answer is 27"),
            ("What is 100 + 200 ?", "[USE_TOOL: calculator expr= 100+200 ] Answer is 300"),
            ("What is 8 + 9 ?", "[USE_TOOL: calculator expr= 8+9 ] Answer is 17"),
        ]
        return random.choice(queries)

    wiki_chunk_size = 50
    w_idx = 0
    
    while w_idx < len(wikitext_tokens):
        # 1. Append a chunk of real wikitext
        chunk = wikitext_tokens[w_idx:w_idx + wiki_chunk_size]
        fused_tokens.extend(chunk)
        w_idx += wiki_chunk_size
        
        # 2. Interleave a capability task
        task_type = random.randint(0, 3)
        if task_type == 0:
            prompt, target = random.choice(translations)
        elif task_type == 1:
            prompt, target = gen_reasoning()
        elif task_type == 2:
            prompt, target = random.choice(code_templates)
        else:
            prompt, target = gen_math()
            
        fused_tokens.extend(prompt.split())
        fused_tokens.extend(target.split())
        
    # Filter only tokens in vocab
    valid_tokens = [tok for token in fused_tokens for tok in [token] if tok in tok2id]
    return valid_tokens
