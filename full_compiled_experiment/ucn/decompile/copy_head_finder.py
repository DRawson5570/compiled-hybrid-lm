from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class CopyHeadCandidate:
    layer: int
    head: int
    prev_token_attention: float
    copy_strength: float


def find_copy_heads(
    collector,
    n_top: int = 10,
    max_texts: int = 50,
    max_length: int = 64,
) -> List[CopyHeadCandidate]:
    texts = _get_wikitext_samples()[:max_texts]
    attn_data = collector.collect_attention_from_layer(
        texts,
        layers=list(range(collector.n_layers)),
        max_length=max_length,
    )

    results = []
    n_layers = collector.n_layers
    n_heads = collector.n_heads

    for layer_idx in range(n_layers):
        if layer_idx not in attn_data:
            continue
        patterns_list = attn_data[layer_idx]
        if not patterns_list:
            continue

        for head_idx in range(n_heads):
            prev_attn = 0.0
            total = 0.0
            copy_scores = []

            for pattern in patterns_list:
                if pattern.dim() != 4:
                    continue
                head_pattern = pattern[0, head_idx]
                for pos in range(1, min(head_pattern.shape[0], max_length)):
                    prev_attn += float(head_pattern[pos, pos - 1])
                    total += 1.0

            if total == 0:
                continue

            avg_prev_attn = prev_attn / total

            if avg_prev_attn > 0.0:
                results.append(
                    CopyHeadCandidate(
                        layer=layer_idx,
                        head=head_idx,
                        prev_token_attention=avg_prev_attn,
                        copy_strength=avg_prev_attn,
                    )
                )

    results.sort(key=lambda x: x.copy_strength, reverse=True)
    return results[:n_top]


def measure_copy_fidelity(
    collector,
    layer: int,
    head: int,
    n_texts: int = 20,
    max_length: int = 64,
) -> Dict[str, float]:
    texts = _get_wikitext_samples()[:n_texts]

    attn_data = collector.collect_attention_from_layer(
        texts,
        layers=[layer],
        max_length=max_length,
    )

    patterns = attn_data.get(layer, [])
    if not patterns:
        return {"prev_attention": 0.0, "diag_attention": 0.0}

    prev_attn_sum = 0.0
    diag_attn_sum = 0.0
    total_positions = 0.0

    for pattern in patterns:
        if pattern.dim() != 4:
            continue
        head_pat = pattern[0, head]
        for pos in range(1, min(head_pat.shape[0], max_length)):
            prev_attn_sum += float(head_pat[pos, pos - 1])
            if pos < head_pat.shape[0]:
                diag_attn_sum += float(head_pat[pos, pos])
            total_positions += 1.0

    return {
        "prev_attention": prev_attn_sum / max(total_positions, 1.0),
        "diag_attention": diag_attn_sum / max(total_positions, 1.0),
    }


def run_copy_probe(
    collector,
    layer: int,
    head: int,
    n_texts: int = 10,
) -> Dict[str, float]:
    texts = _get_wikitext_samples()[:n_texts]

    logits_list = []
    hidden_list = []
    for text in texts:
        logits, hiddens = collector.run_model_with_output_hidden(text)
        logits_list.append(logits)
        hidden_list.append(hiddens)

    return {"layer": layer, "head": head, "n_texts": n_texts}


def _get_wikitext_samples() -> List[str]:
    return [
        "The cat sat on the mat and looked around the room.",
        "Machine learning is a field of artificial intelligence that enables computers to learn from data.",
        "The capital of France is Paris, a city known for its art and culture.",
        "Python is a high-level programming language used for web development and data science.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Neural networks consist of layers of interconnected nodes that process information.",
        "The Earth orbits the Sun at an average distance of about 93 million miles.",
        "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
        "Water boils at 100 degrees Celsius and freezes at 0 degrees Celsius.",
        "The human brain contains approximately 86 billion neurons connected by synapses.",
        "Einstein developed the theory of relativity which revolutionized physics.",
        "The Amazon rainforest produces about 20% of the world's oxygen supply.",
        "Deep learning models require large amounts of data and computational resources.",
        "The Great Wall of China is over 13,000 miles long and was built over centuries.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "JavaScript is commonly used for front-end web development alongside HTML and CSS.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "Climate change poses significant risks to ecosystems and human societies worldwide.",
        "DNA contains the genetic instructions for the development of all living organisms.",
        "Blockchain technology enables decentralized and secure digital transactions.",
        "The Roman Empire at its peak controlled territories across three continents.",
        "Quantum computing uses principles of quantum mechanics to process information.",
        "The piano has 88 keys and is one of the most popular musical instruments.",
        "Mitochondria are often called the powerhouse of the cell for producing energy.",
        "The Industrial Revolution began in Britain in the late 18th century.",
        "Artificial neural networks were inspired by the biological structure of the brain.",
        "The Nile River is the longest river in the world flowing through multiple countries.",
        "Fibonacci numbers appear frequently in nature from flower petals to spiral shells.",
        "The Unix operating system was developed at Bell Labs in the early 1970s.",
        "Gravity is the force that attracts objects with mass toward each other.",
        "The printing press was invented by Johannes Gutenberg in the 15th century.",
        "Helium is the second lightest element and is used in balloons and airships.",
        "The periodic table organizes chemical elements by atomic number and properties.",
        "Renaissance art flourished in Italy during the 14th through 17th centuries.",
        "TCP/IP is the fundamental protocol suite that powers the modern internet.",
        "The moon's gravitational pull causes tides in Earth's oceans.",
        "Game theory studies strategic interactions where the outcome depends on choices of all participants.",
        "The binary number system uses only two digits, 0 and 1, to represent all values.",
        "Photos provide a way to capture and preserve visual memories over time.",
        "Bananas are a good source of potassium and are one of the most consumed fruits.",
        "The first successful airplane flight was achieved by the Wright brothers in 1903.",
        "Vitamin C is essential for the growth and repair of tissues in the human body.",
        "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
        "The theory of evolution by natural selection was proposed by Charles Darwin.",
        "Cloud computing allows users to access computing resources over the internet.",
        "The human heart beats approximately 100,000 times per day on average.",
        "Chess is a two-player strategy board game that dates back over 1500 years.",
        "The speed of sound is about 343 meters per second in dry air at 20 degrees Celsius.",
        "Diamonds are formed under high pressure and temperature conditions deep within the Earth.",
        "The first electronic computer ENIAC was built during World War II.",
    ]
