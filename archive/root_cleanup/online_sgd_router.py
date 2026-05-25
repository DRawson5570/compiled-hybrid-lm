"""online_sgd_router.py

Implements dynamic online residual SGD tuning for CMI blenders during live inference.
Runs small stochastic gradient descent backpropagation steps to adapt routing
gains to immediate feedback constraints, strictly in a sandboxed, low-memory shape.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from hybrid.v3_super_blender.model import CausalConvBlender

def run_online_sgd_step():
    print("=" * 80)
    print("         CMI ONLINE ROUTER GRADO-RESIDUAL SGD TUNING WORKFLOW")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate sandboxed blender model
    model = CausalConvBlender(
        in_dim=32, n_channels=4, channels=64, kernel_size=3, num_layers=2, dropout=0.0
    ).to(device)
    model.train() # Enable training state for backpropagation
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("Initialized SGD optimizer with fast learning-rate (lr=0.01, momentum=0.9).")
    
    # Simulate an immediate reward constraint (e.g. user requested higher InstructChannel focus)
    # Feature shape is (B, SeqLen, FeatDim) = (1, 5, 32)
    features = torch.randn(1, 5, 32, device=device, requires_grad=True)
    
    # Target allocations (e.g. step 5 requires 100% InstructChannel route focus)
    target_weights = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    
    print("\nExecuting forward pass...")
    # Forward through causal convolutional network
    log_w = model(features).squeeze(0) # (SeqLen, C)
    latest_w = log_w[-1].exp() # (C,)
    
    print(f"  Pre-SGD weight allocations: {latest_w.cpu().detach().numpy()}")
    
    # Loss formulation (Mean Squared Error or Negative Log Likelihood distance to optimal routing)
    loss = nn.functional.mse_loss(latest_w.unsqueeze(0), target_weights)
    print(f"Calculated MSE distance loss: {loss.item():.6f}")
    
    # 2. Dynamic Update Backpropagation
    print("\nExecuting backward pass and parameter update...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Confirm optimization adjusted weight metrics correctly
    model.eval()
    with torch.no_grad():
        new_log_w = model(features).squeeze(0)
        new_w = new_log_w[-1].exp()
    print(f"  Post-SGD weight allocations: {new_w.cpu().numpy()}")
    print(f"  Did weight align closer to target? {'Yes! (Pass)' if new_w[0] > latest_w[0] else 'No (Fail)'}")
    print("=" * 80)

if __name__ == "__main__":
    run_online_sgd_step()
