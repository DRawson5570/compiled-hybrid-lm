"""hybrid/v3_super_blender/train_and_eval_v33.py

Automated script to post-process v33 feature dumps, train and evaluate
all 4 sequence-aware routing blenders on v33 features.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CHANNEL_NAMES = [
    "kn", "mix",
    "tri_f", "tri_s", "bi_f", "bi_s", "uc_f", "uc_s",
    "attn_uf", "attn_us", "attn_ug",
    "attn_rf1", "attn_rs1",
    "attn_rf2", "attn_rs2", "attn_rg2",
    "attn_rf3", "attn_rs3",
    "ppmi", "knn", "shape"
]


def ensure_channel_names(npz_path: Path):
    if not npz_path.exists():
        print(f"File {npz_path} does not exist yet!")
        return False
    
    try:
        data = dict(np.load(npz_path, allow_pickle=True))
        modified = False
        
        if "channel_names" not in data or data["channel_names"] is None:
            print(f"Adding channel_names to {npz_path}...")
            data["channel_names"] = CHANNEL_NAMES
            modified = True
            
        if "targets" not in data:
            print(f"Adding targets (copied from observed) to {npz_path}...")
            data["targets"] = data["observed"]
            modified = True
            
        if modified:
            np.savez_compressed(npz_path, **data)
            print(f"Successfully updated {npz_path}")
        else:
            print(f"{npz_path} is healthy.")
        return True
    except Exception as e:
        print(f"Error checking/updating {npz_path}: {e}")
        return False


def run_command(cmd_args: list[str]) -> int:
    print(f"\n>>> Running: {' '.join(cmd_args)}")
    res = subprocess.run(cmd_args, cwd=str(REPO))
    return res.returncode


def main():
    data_dir = REPO / "hybrid/v3_super_blender/data_real_v33"
    val_path = data_dir / "val.npz"
    eval_path = data_dir / "eval.npz"

    print("Post-processing and checking NPZ files...")
    ensure_channel_names(val_path)
    ensure_channel_names(eval_path)

    python_bin = sys.executable
    print(f"Using Python: {python_bin}")

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
            "--data-dir", str(data_dir),
            "--out-dir", "hybrid/v3_super_blender/saved_models_v33",
            "--epochs", "100",
            "--device", "cuda"
        ]
        code = run_command(cmd)
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
            "--blender", f"hybrid/v3_super_blender/saved_models_v33/blender_{m}.pt",
            "--data-dir", str(data_dir),
            "--out", f"hybrid/v3_super_blender/data_real_v33/eval_report_{m}.json",
            "--slice", "eval",
            "--device", "cuda"
        ]
        code = run_command(cmd)
        if code != 0:
            print(f"Error evaluating model {m}")
            sys.exit(code)

    print("\nAll models trained and evaluated successfully!")


if __name__ == "__main__":
    main()