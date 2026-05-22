import json
from pathlib import Path

def main():
    data_dir = Path("hybrid/v3_super_blender/data_real")
    models = ["window_mlp", "lookback_mlp", "gru", "causal_conv"]
    
    print("# Super Blender Performance Summary")
    print("| Model Type | Trained Blender PPL | Uniform Mix PPL | Best Single Channel | Best Single PPL | Oracle PPL (Lower Bound) |")
    print("|---|---|---|---|---|---|")
    
    for m in models:
        path = data_dir / f"eval_report_{m}.json"
        if not path.exists():
            print(f"| {m} | N/A (Missing JSON) | - | - | - | - |")
            continue
            
        with open(path, "r") as f:
            r = json.load(f)
            
        tb_ppl = f"**{r['trained_blender_ppl']:.3f}**" if r['trained_blender_ppl'] < 29.0 else f"{r['trained_blender_ppl']:.3f}"
        print(f"| {r['model_type']} | {tb_ppl} | {r['uniform_mix_ppl']:.3f} | {r['best_single_channel']} | {r['best_single_ppl']:.3f} | {r['oracle_per_token_ppl']:.3f} |")

if __name__ == "__main__":
    main()
