"""dynamic_entropy_temp.py

Implements Phase 12.1: Dynamic Entropy-Based Temperature Scheduling.
Based on the uncertainty (Shannon entropy) of CMI channel routing predictions,
the generator temperature is dynamically tuned at each autoregressive step.
"""
from __future__ import annotations

import math
import torch

def calculate_shannon_entropy(routing_probs: torch.Tensor) -> float:
    """Calculates the Shannon entropy (in bits) of the routing probabilities.
    For 4 channels, max possible entropy is log2(4) = 2.0 bits.
    """
    entropy = 0.0
    for p in routing_probs.tolist():
        if p > 1e-9:
            entropy -= p * math.log2(p)
    return entropy

def get_adaptive_temperature(entropy: float, base_temp: float = 0.7, min_temp: float = 0.1, max_temp: float = 1.2) -> float:
    """Dynamically scales temperature based on entropy:
    High entropy (high uncertainty) -> Cool down to min_temp for deterministic behavior.
    Low entropy (high confidence) -> Warm up to base_temp or max_temp for natural variety.
    """
    # Max entropy for 4 channels is 2.0
    normalized_uncertainty = min(1.0, max(0.0, entropy / 2.0))
    
    # Invert the relationship: high uncertainty = low temperature.
    # Linear interpolation:
    temp = base_temp + (1.0 - normalized_uncertainty) * (max_temp - base_temp) - (normalized_uncertainty * (base_temp - min_temp))
    return float(min(max_temp, max(min_temp, temp)))

def run_entropy_tests():
    print("=" * 80)
    print("        CMI DYNAMIC ENTROPY-BASED TEMPERATURE SCHEDULER")
    print("=" * 80)
    
    scenarios = [
        ("High Certainty (InstructChannel dominant)", torch.tensor([0.97, 0.01, 0.01, 0.01])),
        ("Moderate Uncertainty (Two contending channels)", torch.tensor([0.48, 0.48, 0.02, 0.02])),
        ("Maximum Uncertainty (Perfect uniform split)", torch.tensor([0.25, 0.25, 0.25, 0.25]))
    ]
    
    for desc, probs in scenarios:
        entropy = calculate_shannon_entropy(probs)
        temp = get_adaptive_temperature(entropy)
        print(f"Scenario: {desc}")
        print(f"  Routing Probabilities: {probs.tolist()}")
        print(f"  Shannon Entropy:       {entropy:.4f} bits (Max: 2.0)")
        print(f"  Adaptive Temperature:  {temp:.4f}")
        print("-" * 50)
        
    print("Step 1 Completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_entropy_tests()
