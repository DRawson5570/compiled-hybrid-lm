"""hybrid/v3_super_blender/summary_table_v33.py

Generate a markdown table summarizing the performance of all 4 sequence-aware
routing blenders trained/evaluated on the v33 features.
"""
from __future__ import annotations

import json
from pathlib import Path

def main():
    # Use repo root relative path
    repo = Path(__file__).resolve().parents[2]
    data_dir = repo / "hybrid/v3_super_blender/data_real_v33"
    models = ["window_mlp", "lookback_mlp", "gru", "causal_conv"]
    
    print("\n# Super Blender v33 (21-Channel) Performance Summary\n")
    print("| Model Type | Trained Blender PPL | Uniform Mix PPL | Best Single Channel | Best Single PPL | Oracle PPL (Lower Bound) |")
    print("|---|---|---|---|---|---|")
    
    for m in models:
        path = data_dir / f"eval_report_{m}.json"
        if not path.exists():
            print(f"| {m:15s} | N/A (Pending)       | - | - | - | - |")
            continue
            
        with open(path, "r") as f:
            r = json.load(f)
            
        tb_ppl = f"**{r['trained_blender_ppl']:.3f}**" if r['trained_blender_ppl'] < 29.0 else f"{r['trained_blender_ppl']:.3f}"
        print(f"| {r['model_type']:15s} | {tb_ppl:18s} | {r['uniform_mix_ppl']:.3f} | {r['best_single_channel']:20s} | {r['best_single_ppl']:.3f} | {r['oracle_per_token_ppl']:.3f} |")
    print()

if __name__ == "__main__":
    main()
