"""hybrid/v3_super_blender/train_and_eval_all.py

Automated script to train and evaluate all 4 sequence-aware routing blenders
on pe2, using a specified GPU.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def run_command(cmd_args: list[str], env_updates: dict[str, str] | None = None) -> int:
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    print(f"\n>>> Running: {' '.join(cmd_args)}")
    res = subprocess.run(cmd_args, env=env, cwd=str(REPO))
    return res.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="2")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    python_bin = sys.executable
    print(f"Using Python: {python_bin}")
    print(f"CUDA_VISIBLE_DEVICES={args.gpu}")

    models = ["window_mlp", "lookback_mlp", "gru", "causal_conv"]
    
    # 1. Train all models
    for m in models:
        print(f"\n========================================================")
        print(f"TRAINING MODEL: {m}")
        print(f"========================================================")
        cmd = [
            python_bin,
            "hybrid/v3_super_blender/train.py",
            "--model-type", m,
            "--data-dir", "hybrid/v3_super_blender/data_real",
            "--epochs", str(args.epochs),
            "--device", "cuda"
        ]
        code = run_command(cmd, {"CUDA_VISIBLE_DEVICES": args.gpu})
        if code != 0:
            print(f"Error training model {m}")
            sys.exit(code)

    # 2. Evaluate all models
    for m in models:
        print(f"\n========================================================")
        print(f"EVALUATING MODEL: {m}")
        print(f"========================================================")
        cmd = [
            python_bin,
            "hybrid/v3_super_blender/eval.py",
            "--blender", f"hybrid/v3_super_blender/saved_models/blender_{m}.pt",
            "--data-dir", "hybrid/v3_super_blender/data_real",
            "--slice", "eval",
            "--device", "cuda"
        ]
        code = run_command(cmd, {"CUDA_VISIBLE_DEVICES": args.gpu})
        if code != 0:
            print(f"Error evaluating model {m}")
            sys.exit(code)

    print("\nAll models trained and evaluated successfully!")

if __name__ == "__main__":
    main()
