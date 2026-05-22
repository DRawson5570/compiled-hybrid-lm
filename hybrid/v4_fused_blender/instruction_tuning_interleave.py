"""hybrid/v4_fused_blender/instruction_tuning_interleave.py

Generates and tokenizes an interleaved Instruct dataset stream (combining translation,
reasoning tasks, code blocks, and standard WikiText BPE sequences) for alignment training.
"""
from __future__ import annotations

import math
import sys
import torch
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hybrid.v4_fused_blender.public_eval_harness import GPTOvocabSim

def get_alpaca_dolly_tasks() -> list[dict[str, str]]:
    """Simulates realistic downstream capability datasets (Alpaca/Dolly templates)."""
    return [
        {"instruction": "Translate the cat jumped over the fence to french", "response": "le chat a sauté par-dessus la barrière"},
        {"instruction": "If A is taller than B and B is taller than C, who is the tallest?", "response": "A"},
        {"instruction": "Write a recursive Fibonacci sequence in Python", "response": "def fib(n): return fib(n-1) + fib(n-2)"},
        {"instruction": "Calculate speed given distance=100m time=10s", "response": "speed = 10 m/s"}
    ]

def interleave_and_tokenize(wikitext_tokens: list[str]) -> list[tuple[list[int], list[int]]]:
    print("[instruction_tuning_interleave] Fetching evaluation and training tokens...")
    tokenizer = GPTOvocabSim()
    tasks = get_alpaca_dolly_tasks()
    
    interleaved_data = []
    
    # 1. First tokenise instructions
    for idx, item in enumerate(tasks):
        inst_ids = tokenizer.encode(item["instruction"])
        resp_ids = tokenizer.encode(item["response"])
        
        # Interleave BPE token chunks from wikitext prose around dataset elements
        start_prose = tokenizer.encode(" ".join(wikitext_tokens[idx*50: (idx+1)*50]))
        
        # Combined context BPE token sequences
        context = start_prose + inst_ids
        target = resp_ids
        interleaved_data.append((context, target))
        
    print(f"[instruction_tuning_interleave] Finished compiling {len(interleaved_data)} interleaved dataset task-streams.")
    return interleaved_data

if __name__ == "__main__":
    dummy_wiki = ["the", "system", "coordinates", "deep", "linguistic", "features", "perfectly"] * 30
    interleave_and_tokenize(dummy_wiki)