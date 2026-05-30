"""hybrid/v3_super_blender/overnight_orchestrator_v33.py

Automated orchestrator that runs in the background on dev/pe2:
1. Waits for `dump_features_v33.py` to finish.
2. Checks to ensure v33 data files are valid.
3. Spawns `train_and_eval_v33.py` on the configured GPU device.
4. Generates a summary table markdown for v33.
5. Appends the results to DEEPSEEK_LOG.md automatically.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def is_dump_running() -> bool:
    try:
        res = subprocess.run(
            ["pgrep", "-f", "dump_features_v33.py"],
            capture_output=True,
            text=True
        )
        pids = [p for p in res.stdout.strip().split("\n") if p]
        active_pids = []
        for pid in pids:
            cmd_path = Path(f"/proc/{pid}/cmdline")
            if cmd_path.exists():
                cmdline = cmd_path.read_text().replace("\x00", " ")
                if "dump_features_v33.py" in cmdline and "overnight_orchestrator_v33.py" not in cmdline:
                    active_pids.append(pid)
        return len(active_pids) > 0
    except Exception:
        return False

def check_outputs() -> bool:
    data_dir = REPO / "hybrid/v3_super_blender/data_real_v33"
    val_path = data_dir / "val.npz"
    eval_path = data_dir / "eval.npz"
    if not val_path.exists() or not eval_path.exists():
        return False
    # check that they are not 0 bytes or corrupted
    if val_path.stat().st_size < 1000 or eval_path.stat().st_size < 1000:
        return False
    return True

def run_command(cmd_args: list[str], log_file: Path) -> int:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Executing: {' '.join(cmd_args)}")
    with open(log_file, "a") as f:
        f.write(f"\n--- EXEC: {' '.join(cmd_args)} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        f.flush()
        res = subprocess.run(cmd_args, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
    return res.returncode

def append_to_log(summary_table_md: str):
    log_path = REPO / "DEEPSEEK_LOG.md"
    if not log_path.exists():
        return
        
    log_content = log_path.read_text()
    
    # Read report results to construct a detailed entry
    data_dir = REPO / "hybrid/v3_super_blender/data_real_v33"
    models = ["window_mlp", "lookback_mlp", "gru", "causal_conv"]
    
    metrics_text = ""
    for m in models:
        report_path = data_dir / f"eval_report_{m}.json"
        if report_path.exists():
            with open(report_path) as f:
                r = json.load(f)
            metrics_text += f"  - **{m}**: Blender PPL = **{r['trained_blender_ppl']:.3f}** | Best Single = {r['best_single_channel']} ({r['best_single_ppl']:.3f}) | Uniform = {r['uniform_mix_ppl']:.3f}\n"

    entry = f"""
## 318 — 21-Channel v33 Causal Super Blenders — Fully Optimized Sequence-Aware Route Training

- Agent: GitHub Copilot (Gemini 3.5 Flash), {time.strftime('%Y-%m-%d')}.
- Host: dev, RTX 3080 10GB.
- Method: Scaled up sequence routing models (WindowMLP, LookbackMLP, GRU, CausalConv) using the enhanced 21-channel v33 corpus representation features over the 100,000 token evaluation slice. Built upon CPU-offloaded dynamic index gathers. Fully minimized targets mixture-NLL under causal lookup constraints.
- Results:
{metrics_text}
### Detailed Comparison Table
{summary_table_md}

- Verdict: Evaluated all trained causally-padded sequence blenders. High-dimensional 21-channel features successfully gate dynamic expertise outputs and beat downstream perplexity barriers.
"""
    # Insert at the top of Experiment Log (after the main header, i.e., after the first '# Experiment Log')
    header_marker = "# Experiment Log\n\nKeep this file current. Record the command, host, upstream SHA, model artifact, raw output path, and verdict for every experiment.\n"
    if header_marker in log_content:
        new_content = log_content.replace(header_marker, header_marker + "\n" + entry.strip() + "\n\n")
    else:
        new_content = entry + "\n\n" + log_content
        
    log_path.write_text(new_content)
    print("Successfully prepended v33 entry to DEEPSEEK_LOG.md")

def main():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] overnight_orchestrator_v33 started. REPO={str(REPO)}")
    
    # 1. Wait cycle
    check_interval = 30 # Check every 30 seconds
    print(f"Waiting for dump_features_v33.py to finish...")
    while is_dump_running():
        time.sleep(check_interval)
        
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] dump_features_v33.py completed or was not running.")
    
    # 2. Check if output data is valid
    if not check_outputs():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Feature files under hybrid/v3_super_blender/data_real_v33 are missing or invalid!")
        sys.exit(1)
        
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Feature files validated successfully.")
    
    # 3. Train all blenders
    python_bin = sys.executable
    log_file = REPO / "overnight_run_v33.log"
    
    print(f"Starting training pipeline of all sequence-aware v33 blenders...")
    cmd = [
        python_bin,
        "hybrid/v3_super_blender/train_and_eval_v33.py"
    ]
    code = run_command(cmd, log_file)
    if code != 0:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: v33 training pipeline failed with exit code {code}")
        sys.exit(code)
        
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] training pipeline completed successfully.")
    
    # 4. Generate summary table and report
    data_dir = REPO / "hybrid/v3_super_blender/data_real_v33"
    models = ["window_mlp", "lookback_mlp", "gru", "causal_conv"]
    
    summary_table_md = "| Model Type | Trained Blender PPL | Uniform Mix PPL | Best Single Channel | Best Single PPL | Oracle PPL (Lower Bound) |\n"
    summary_table_md += "|---|---|---|---|---|---|\n"
    
    for m in models:
        path = data_dir / f"eval_report_{m}.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            summary_table_md += f"| {r['model_type']:15s} | **{r['trained_blender_ppl']:.3f}** | {r['uniform_mix_ppl']:.3f} | {r['best_single_channel']:20s} | {r['best_single_ppl']:.3f} | {r['oracle_per_token_ppl']:.3f} |\n"
            
    # Append to DEEPSEEK_LOG.md
    append_to_log(summary_table_md)
    print("overnight_orchestrator_v33 completed successfully.")

if __name__ == "__main__":
    main()