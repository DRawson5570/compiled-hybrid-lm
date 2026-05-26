#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TARGET="${TARGET:-local-700m}"
MODEL_CONFIG=""
BACKEND=""
DATA_MODE="${DATA_MODE:-c4-mix}"
C4_RATIO="${C4_RATIO:-1.0}"
TRAIN_SURFACE="${TRAIN_SURFACE:-}"
EPOCHS="${EPOCHS:-}"
TARGET_EPOCH="${TARGET_EPOCH:-1000}"
STEPS=""
BATCH="${BATCH:-}"
SEQ_LEN="${SEQ_LEN:-}"
EVAL_TOKENS="${EVAL_TOKENS:-}"
LR="${LR:-}"
STEERER_LR="${STEERER_LR:-}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-}"
DISABLE_STEERER_AFTER_PLATEAU="${DISABLE_STEERER_AFTER_PLATEAU:-${DISABLE_PRIOR_AFTER_ON_PLATEAU:-}}"
FREEZE_MODEL_UNTIL_STEERER_PPL="${FREEZE_MODEL_UNTIL_STEERER_PPL:-${FREEZE_MODEL_UNTIL_PRIOR_ON_PPL:-}}"
STEERER_WARMUP_PATIENCE="${STEERER_WARMUP_PATIENCE:-${PRIOR_ON_WARMUP_PATIENCE:-}}"
CHECKPOINT="${CHECKPOINT:-}"
INIT_STEERER_CHECKPOINT="${INIT_STEERER_CHECKPOINT:-}"
FREEZE_STEERER=0
CALIBRATE_STEERING_CONTROLS=0
STEERING_CONTROL_STEPS="${STEERING_CONTROL_STEPS:-}"
STEERING_CONTROL_LR="${STEERING_CONTROL_LR:-}"
OUT_DIR="${OUT_DIR:-}"
OUT_DIR_EXPLICIT=0
PORT=""
GPUS=""
ZEROQ_PATH="${ZEROQ_PATH:-$HOME/ZeroQ}"
TORCHRUN="${TORCHRUN:-$ROOT_DIR/.venv/bin/torchrun}"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-artifacts/logs}"
REMOTE_HOST="${REMOTE_HOST:-pe2}"
REMOTE_REPO="${REMOTE_REPO:-/home/drawson/deepseek_experiments}"
REMOTE_TORCHRUN="${REMOTE_TORCHRUN:-/home/drawson/local_venvs/m40_env/bin/torchrun}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/drawson/local_venvs/m40_env/bin/python}"
HF_CACHE="${HF_CACHE:-/mnt/ssd-pgu3/hf_cache}"
FOREGROUND=0
FRESH=0
FORCE_KILL=0
DRY_RUN=0
STATUS_ONLY=0
SYNC=1
ALLOW_EXISTING_OUT_DIR=0
MAX_WARMUP_EPOCHS="${MAX_WARMUP_EPOCHS:-}"
STOP_AFTER_STEERER_WARMUP=0
NO_STEERER_WARMUP=0

usage() {
  cat <<'USAGE'
Usage: ./launch_training.sh [options]

Single launcher for hybrid/train_4b_distributed.py. The Python trainer name is
historical; model size is selected with --model-config.

Targets:
  --target local-700m     Local RTX 3080 700M dense C4-mix run. Default.
  --target local-700m-baseline
                          Full neural 700M dense baseline, no compiled steerer.
  --target local-700m-thesis
                          Full neural 700M dense thesis run with steerer warmup.
  --target local-700m-baseline-zeroq
                          Local RTX 3080 700M ZeroQ run, train top neural layers only.
  --target local-700m-thesis-zeroq
                          Local RTX 3080 700M ZeroQ thesis run with steerer warmup.
  --target pe2-4b         pe2 GPU1 4B ZeroQ 4-bit C4-mix run.
  --target pe2-700m-baseline-zeroq
                          pe2 GPU1 700M ZeroQ run, train top neural layers only.
  --target pe2-700m-thesis-zeroq
                          pe2 GPU1 700M ZeroQ thesis run with steerer warmup.

Common options:
  --fresh                 Start from scratch instead of resuming a checkpoint.
  --epochs N              Main neural-training epochs to run after warmup. Local default is target - main checkpoint epoch.
  --target-epoch N        Local target absolute main epoch when --epochs is omitted. Default: 1000
  --model-config NAME     Override target model config, e.g. 700m or 4b.
  --backend MODE          dense or zeroq.
  --train-surface NAME    Override trainable surface, e.g. cmi_steerer, top2, top2_cmi_steerer.
  --data-mode MODE        wikitext or c4-mix. Default: c4-mix
  --c4-ratio X            C4 mixing ratio. Default: 1.0
  --checkpoint PATH       Checkpoint path on the machine where training runs.
  --init-steerer-checkpoint PATH
                          Load only steerer/cartridge weights from PATH; do not resume model or counters.
  --freeze-steerer       Keep the loaded steerer/cartridge active but freeze its parameters.
  --calibrate-steering-controls
                          Freeze model/cartridge body, overfit one batch to solve alpha/beta/gamma, then freeze steerer.
  --steering-control-steps N
                          Repeated-batch alpha/beta/gamma calibration steps. Defaults to --steps.
  --steering-control-lr X
                          Alpha/beta/gamma calibration learning rate. Default: trainer default.
  --out-dir PATH          Output directory on the machine where training runs.
  --allow-existing-out-dir
                          Allow --fresh to write into an output dir that already has checkpoints.
  --steps N               Steps per epoch.
  --batch N               Batch size. Target default: local-700m=1, pe2-4b=4
  --seq-len N             Sequence length. Target default: local-700m=512, pe2-4b=128
  --eval-tokens N         Eval tokens. Target default: local-700m=8192, pe2-4b=2048
  --lr X                  Model-surface learning rate. Default: 1e-4
  --steerer-lr X          Steerer learning rate. Default: 1e-5
  --early-stop-metric M   none, steered, blind, or either. Default: steered
  --early-stop-patience N Epochs without improvement before stopping. Default: 40
  --disable-steerer-after-plateau N
                          Stop using the steerer during training after steerer-on eval is stale for N main epochs.
  --freeze-model-until-steerer-ppl X
                          Train only the steerer until eval_steerer_on <= X. Warmup epochs do not count against --epochs.
  --steerer-warmup-patience N
                          Consecutive steerer-on evals under threshold before neural training starts. Default: 1
  --max-warmup-epochs N   Stop if the steerer gate has not opened after N warmup epochs.
  --stop-after-steerer-warmup
                          Stop immediately after the steerer reaches the warmup threshold; useful for cartridge calibration.
  --no-steerer-warmup    Do not freeze the neural surface before main training. Use with --init-steerer-checkpoint
                          when a calibrated per-cartridge steerer is already useful.
  --port N                torchrun master port.
  --gpus LIST             CUDA_VISIBLE_DEVICES list.
  --status                Show active matching training processes and exit.
  --force-kill            Stop active matching processes before launching.
  --dry-run               Print command without launching.
  --foreground            Run attached instead of detached nohup.

Remote pe2 options:
  --remote-host HOST      Default: pe2
  --remote-repo PATH      Default: /home/drawson/deepseek_experiments
  --remote-torchrun PATH  Default: /home/drawson/local_venvs/m40_env/bin/torchrun
  --remote-python PATH    Default: /home/drawson/local_venvs/m40_env/bin/python
  --hf-cache PATH         Default: /mnt/ssd-pgu3/hf_cache
  --no-sync               Do not rsync trainer/backend files before remote launch.

Examples:
  ./launch_training.sh --target local-700m --status
  ./launch_training.sh --target local-700m --epochs 120 --force-kill
  ./launch_training.sh --target pe2-4b --epochs 20 --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --target-epoch) TARGET_EPOCH="$2"; shift 2 ;;
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --train-surface) TRAIN_SURFACE="$2"; shift 2 ;;
    --data-mode) DATA_MODE="$2"; shift 2 ;;
    --c4-ratio) C4_RATIO="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --init-steerer-checkpoint) INIT_STEERER_CHECKPOINT="$2"; shift 2 ;;
    --freeze-steerer) FREEZE_STEERER=1; shift ;;
    --calibrate-steering-controls) CALIBRATE_STEERING_CONTROLS=1; shift ;;
    --steering-control-steps) STEERING_CONTROL_STEPS="$2"; shift 2 ;;
    --steering-control-lr) STEERING_CONTROL_LR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; OUT_DIR_EXPLICIT=1; shift 2 ;;
    --allow-existing-out-dir) ALLOW_EXISTING_OUT_DIR=1; shift ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --eval-tokens) EVAL_TOKENS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --steerer-lr) STEERER_LR="$2"; shift 2 ;;
    --early-stop-metric) EARLY_STOP_METRIC="$2"; shift 2 ;;
    --early-stop-patience) EARLY_STOP_PATIENCE="$2"; shift 2 ;;
    --disable-steerer-after-plateau|--disable-prior-after-on-plateau) DISABLE_STEERER_AFTER_PLATEAU="$2"; shift 2 ;;
    --freeze-model-until-steerer-ppl|--freeze-model-until-prior-on-ppl) FREEZE_MODEL_UNTIL_STEERER_PPL="$2"; shift 2 ;;
    --steerer-warmup-patience|--prior-on-warmup-patience) STEERER_WARMUP_PATIENCE="$2"; shift 2 ;;
    --max-warmup-epochs) MAX_WARMUP_EPOCHS="$2"; shift 2 ;;
    --stop-after-steerer-warmup) STOP_AFTER_STEERER_WARMUP=1; shift ;;
    --no-steerer-warmup) NO_STEERER_WARMUP=1; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --zeroq-path) ZEROQ_PATH="$2"; shift 2 ;;
    --torchrun) TORCHRUN="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --remote-host) REMOTE_HOST="$2"; shift 2 ;;
    --remote-repo) REMOTE_REPO="$2"; shift 2 ;;
    --remote-torchrun) REMOTE_TORCHRUN="$2"; shift 2 ;;
    --remote-python) REMOTE_PYTHON="$2"; shift 2 ;;
    --hf-cache) HF_CACHE="$2"; shift 2 ;;
    --no-sync) SYNC=0; shift ;;
    --status) STATUS_ONLY=1; shift ;;
    --force-kill) FORCE_KILL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$TARGET" in
  local-700m)
    MODEL_CONFIG="${MODEL_CONFIG:-700m}"
    BACKEND="${BACKEND:-dense}"
    TRAIN_SURFACE="${TRAIN_SURFACE:-cmi_steerer}"
    STEPS="${STEPS:-240}"
    BATCH="${BATCH:-1}"
    SEQ_LEN="${SEQ_LEN:-512}"
    EVAL_TOKENS="${EVAL_TOKENS:-8192}"
    LR="${LR:-1e-4}"
    STEERER_LR="${STEERER_LR:-1e-5}"
    EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-steered}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
    DISABLE_STEERER_AFTER_PLATEAU="${DISABLE_STEERER_AFTER_PLATEAU:-0}"
    FREEZE_MODEL_UNTIL_STEERER_PPL="${FREEZE_MODEL_UNTIL_STEERER_PPL:-50}"
    STEERER_WARMUP_PATIENCE="${STEERER_WARMUP_PATIENCE:-1}"
    PORT="${PORT:-29583}"
    GPUS="${GPUS:-0}"
    OUT_DIR="${OUT_DIR:-artifacts/train_700m_cmi_steerer_dense_c4_mix_seq512_20260526}"
    ;;
  local-700m-baseline)
    MODEL_CONFIG="${MODEL_CONFIG:-700m}"
    BACKEND="${BACKEND:-dense}"
    TRAIN_SURFACE="${TRAIN_SURFACE:-full}"
    STEPS="${STEPS:-120}"
    BATCH="${BATCH:-1}"
    SEQ_LEN="${SEQ_LEN:-128}"
    EVAL_TOKENS="${EVAL_TOKENS:-4096}"
    LR="${LR:-3e-5}"
    STEERER_LR="${STEERER_LR:-0}"
    EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-blind}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
    DISABLE_STEERER_AFTER_PLATEAU="${DISABLE_STEERER_AFTER_PLATEAU:-0}"
    FREEZE_MODEL_UNTIL_STEERER_PPL="${FREEZE_MODEL_UNTIL_STEERER_PPL:-}"
    STEERER_WARMUP_PATIENCE="${STEERER_WARMUP_PATIENCE:-1}"
    PORT="${PORT:-29584}"
    GPUS="${GPUS:-0}"
    OUT_DIR="${OUT_DIR:-artifacts/train_700m_full_dense_c4_mix_baseline_20260526}"
    ;;
  local-700m-thesis)
    MODEL_CONFIG="${MODEL_CONFIG:-700m}"
    BACKEND="${BACKEND:-dense}"
    TRAIN_SURFACE="${TRAIN_SURFACE:-full_cmi_steerer}"
    STEPS="${STEPS:-120}"
    BATCH="${BATCH:-1}"
    SEQ_LEN="${SEQ_LEN:-128}"
    EVAL_TOKENS="${EVAL_TOKENS:-4096}"
    LR="${LR:-3e-5}"
    STEERER_LR="${STEERER_LR:-1e-4}"
    EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-either}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
    DISABLE_STEERER_AFTER_PLATEAU="${DISABLE_STEERER_AFTER_PLATEAU:-0}"
    FREEZE_MODEL_UNTIL_STEERER_PPL="${FREEZE_MODEL_UNTIL_STEERER_PPL:-50}"
    STEERER_WARMUP_PATIENCE="${STEERER_WARMUP_PATIENCE:-1}"
    PORT="${PORT:-29585}"
    GPUS="${GPUS:-0}"
    OUT_DIR="${OUT_DIR:-artifacts/train_700m_full_cmi_steerer_dense_c4_mix_thesis_20260526}"
    ;;
  local-700m-baseline-zeroq)
    MODEL_CONFIG="${MODEL_CONFIG:-700m}"
    BACKEND="${BACKEND:-zeroq}"
    TRAIN_SURFACE="${TRAIN_SURFACE:-top2}"
    STEPS="${STEPS:-80}"
    BATCH="${BATCH:-1}"
    SEQ_LEN="${SEQ_LEN:-128}"
    EVAL_TOKENS="${EVAL_TOKENS:-4096}"
    LR="${LR:-3e-5}"
    STEERER_LR="${STEERER_LR:-0}"
    EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-blind}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
    DISABLE_STEERER_AFTER_PLATEAU="${DISABLE_STEERER_AFTER_PLATEAU:-0}"
    FREEZE_MODEL_UNTIL_STEERER_PPL="${FREEZE_MODEL_UNTIL_STEERER_PPL:-}"
    STEERER_WARMUP_PATIENCE="${STEERER_WARMUP_PATIENCE:-1}"
    PORT="${PORT:-29588}"
    GPUS="${GPUS:-0}"
    OUT_DIR="${OUT_DIR:-artifacts/train_700m_top2_zeroq_4bit_c4_mix_baseline_20260526_3080}"
    ;;
  local-700m-thesis-zeroq)
    MODEL_CONFIG="${MODEL_CONFIG:-700m}"
    BACKEND="${BACKEND:-zeroq}"
    TRAIN_SURFACE="${TRAIN_SURFACE:-top2_cmi_steerer}"
    STEPS="${STEPS:-80}"
    BATCH="${BATCH:-1}"
    SEQ_LEN="${SEQ_LEN:-128}"
    EVAL_TOKENS="${EVAL_TOKENS:-4096}"
    LR="${LR:-3e-5}"
    STEERER_LR="${STEERER_LR:-1e-4}"
    EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-either}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
    DISABLE_STEERER_AFTER_PLATEAU="${DISABLE_STEERER_AFTER_PLATEAU:-5}"
    FREEZE_MODEL_UNTIL_STEERER_PPL="${FREEZE_MODEL_UNTIL_STEERER_PPL:-50}"
    STEERER_WARMUP_PATIENCE="${STEERER_WARMUP_PATIENCE:-1}"
    PORT="${PORT:-29589}"
    GPUS="${GPUS:-0}"
    OUT_DIR="${OUT_DIR:-artifacts/train_700m_top2_cmi_steerer_zeroq_4bit_c4_mix_thesis_20260526_3080}"
    ;;
  pe2-4b)
    MODEL_CONFIG="${MODEL_CONFIG:-4b}"
    BACKEND="${BACKEND:-zeroq}"
    TRAIN_SURFACE="${TRAIN_SURFACE:-cmi_steerer}"
    STEPS="${STEPS:-50}"
    BATCH="${BATCH:-4}"
    SEQ_LEN="${SEQ_LEN:-128}"
    EVAL_TOKENS="${EVAL_TOKENS:-2048}"
    LR="${LR:-1e-4}"
    STEERER_LR="${STEERER_LR:-1e-5}"
    EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-steered}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
    DISABLE_STEERER_AFTER_PLATEAU="${DISABLE_STEERER_AFTER_PLATEAU:-0}"
    FREEZE_MODEL_UNTIL_STEERER_PPL="${FREEZE_MODEL_UNTIL_STEERER_PPL:-50}"
    STEERER_WARMUP_PATIENCE="${STEERER_WARMUP_PATIENCE:-1}"
    PORT="${PORT:-29569}"
    GPUS="${GPUS:-1}"
    OUT_DIR="${OUT_DIR:-artifacts/train_4b_cmi_steerer_zeroq_4bit_c4_mix_20260526_offline_gpu1}"
    ;;
  pe2-700m-baseline-zeroq)
    MODEL_CONFIG="${MODEL_CONFIG:-700m}"
    BACKEND="${BACKEND:-zeroq}"
    TRAIN_SURFACE="${TRAIN_SURFACE:-top2}"
    STEPS="${STEPS:-80}"
    BATCH="${BATCH:-2}"
    SEQ_LEN="${SEQ_LEN:-128}"
    EVAL_TOKENS="${EVAL_TOKENS:-4096}"
    LR="${LR:-3e-5}"
    STEERER_LR="${STEERER_LR:-0}"
    EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-blind}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
    DISABLE_STEERER_AFTER_PLATEAU="${DISABLE_STEERER_AFTER_PLATEAU:-0}"
    FREEZE_MODEL_UNTIL_STEERER_PPL="${FREEZE_MODEL_UNTIL_STEERER_PPL:-}"
    STEERER_WARMUP_PATIENCE="${STEERER_WARMUP_PATIENCE:-1}"
    PORT="${PORT:-29586}"
    GPUS="${GPUS:-1}"
    OUT_DIR="${OUT_DIR:-artifacts/train_700m_top2_zeroq_4bit_c4_mix_baseline_20260526_gpu1}"
    ;;
  pe2-700m-thesis-zeroq)
    MODEL_CONFIG="${MODEL_CONFIG:-700m}"
    BACKEND="${BACKEND:-zeroq}"
    TRAIN_SURFACE="${TRAIN_SURFACE:-top2_cmi_steerer}"
    STEPS="${STEPS:-80}"
    BATCH="${BATCH:-2}"
    SEQ_LEN="${SEQ_LEN:-128}"
    EVAL_TOKENS="${EVAL_TOKENS:-4096}"
    LR="${LR:-3e-5}"
    STEERER_LR="${STEERER_LR:-1e-4}"
    EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-either}"
    EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
    DISABLE_STEERER_AFTER_PLATEAU="${DISABLE_STEERER_AFTER_PLATEAU:-5}"
    FREEZE_MODEL_UNTIL_STEERER_PPL="${FREEZE_MODEL_UNTIL_STEERER_PPL:-50}"
    STEERER_WARMUP_PATIENCE="${STEERER_WARMUP_PATIENCE:-1}"
    PORT="${PORT:-29587}"
    GPUS="${GPUS:-1}"
    OUT_DIR="${OUT_DIR:-artifacts/train_700m_top2_cmi_steerer_zeroq_4bit_c4_mix_thesis_20260526_gpu1}"
    ;;
  *) echo "Unknown target: $TARGET" >&2; usage >&2; exit 2 ;;
esac

if [[ "$NO_STEERER_WARMUP" == "1" ]]; then
  FREEZE_MODEL_UNTIL_STEERER_PPL=""
fi

find_matching_pids_py='import os, sys
model_config = sys.argv[1]
out_dir = sys.argv[2]
matches = []
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid == os.getpid():
        continue
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        continue
    parts = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
    joined = " ".join(parts)
    if "hybrid/train_4b_distributed.py" not in joined and "train_4b_distributed.py" not in joined:
        continue
    has_config = any(parts[i] == "--model-config" and i + 1 < len(parts) and parts[i + 1] == model_config for i in range(len(parts)))
    has_out = (not out_dir) or any(parts[i] == "--out-dir" and i + 1 < len(parts) and parts[i + 1] == out_dir for i in range(len(parts)))
    if has_config and has_out:
        matches.append((pid, joined))
for pid, joined in matches:
    print(f"{pid}\t{joined}")'

find_matching_pids() {
  "$PYTHON" -c "$find_matching_pids_py" "$MODEL_CONFIG" "$OUT_DIR"
}

stop_matching_pids() {
  local matches remaining
  matches="$1"
  if [[ -z "$matches" ]]; then
    return 0
  fi
  echo "$matches" | cut -f1 | while read -r pid; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for _ in {1..10}; do
    remaining="$(find_matching_pids || true)"
    [[ -z "$remaining" ]] && return 0
    sleep 1
  done
  remaining="$(find_matching_pids || true)"
  if [[ -n "$remaining" ]]; then
    echo "$remaining" | cut -f1 | while read -r pid; do
      [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
    done
  fi
}

checkpoint_epoch() {
  local python_bin="$1"
  local checkpoint_path="$2"
  "$python_bin" - "$checkpoint_path" "$MODEL_CONFIG" <<'PY'
import sys, torch
path, expected_model = sys.argv[1], sys.argv[2]
ckpt = torch.load(path, map_location='cpu', weights_only=False)
print(int(ckpt.get('epoch', 0) or 0))
print(
    f"checkpoint={path} model_config={ckpt.get('model_config')} surface={ckpt.get('train_surface')} "
    f"backend={ckpt.get('backend')} eval_s={ckpt.get('eval_s')} eval_b={ckpt.get('eval_b')}",
    file=sys.stderr,
)
if ckpt.get('model_config') != expected_model:
    raise SystemExit(f"checkpoint model_config is not {expected_model}")
PY
}

run_local() {
  if [[ ! -x "$TORCHRUN" ]]; then echo "Error: torchrun not executable: $TORCHRUN" >&2; exit 2; fi
  if [[ ! -x "$PYTHON" ]]; then echo "Error: python not executable: $PYTHON" >&2; exit 2; fi

  local matches checkpoint_epoch_value resume_args log ts
  matches="$(find_matching_pids || true)"
  if [[ "$STATUS_ONLY" == "1" ]]; then
    [[ -n "$matches" ]] && echo "$matches" || echo "No active $MODEL_CONFIG training process found."
    return 0
  fi
  if [[ -n "$matches" && "$FORCE_KILL" != "1" ]]; then
    echo "Refusing to launch because an active $MODEL_CONFIG run was found:" >&2
    echo "$matches" >&2
    echo "Use --force-kill only when you intentionally want to stop and replace it." >&2
    exit 3
  fi
  if [[ -n "$matches" ]]; then
    stop_matching_pids "$matches"
  fi

  if [[ "$FRESH" == "1" && "$ALLOW_EXISTING_OUT_DIR" != "1" ]]; then
    for existing_checkpoint in "$OUT_DIR/best_s.pt" "$OUT_DIR/best.pt" "$OUT_DIR/best_b.pt"; do
      if [[ -e "$existing_checkpoint" ]]; then
        echo "Error: --fresh would overwrite existing checkpoint: $existing_checkpoint" >&2
        echo "Use --out-dir for a new experiment directory, or --allow-existing-out-dir if overwrite is intentional." >&2
        exit 2
      fi
    done
  fi

  checkpoint_epoch_value=0
  resume_args=()
  if [[ "$FRESH" != "1" ]]; then
    if [[ -z "$CHECKPOINT" ]]; then
      for candidate in "$OUT_DIR/best_s.pt" "$OUT_DIR/best.pt" "$OUT_DIR/best_b.pt"; do
        if [[ -f "$candidate" ]]; then CHECKPOINT="$candidate"; break; fi
      done
    fi
    if [[ ! -f "$CHECKPOINT" ]]; then echo "Error: checkpoint not found: $CHECKPOINT" >&2; exit 2; fi
    checkpoint_epoch_value="$(checkpoint_epoch "$PYTHON" "$CHECKPOINT")"
    resume_args=(--resume-checkpoint "$CHECKPOINT")
  fi

  if [[ -z "$EPOCHS" ]]; then
    if [[ "$FRESH" == "1" ]]; then EPOCHS="$TARGET_EPOCH"; else EPOCHS=$(( TARGET_EPOCH - checkpoint_epoch_value )); fi
  fi
  if (( EPOCHS <= 0 )); then echo "Error: computed --epochs $EPOCHS. Pass --epochs N to continue." >&2; exit 2; fi

  mkdir -p "$LOG_DIR"
  ts="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/train_${MODEL_CONFIG}_${BACKEND}_${TARGET}_${ts}.log"

  local cmd=(
    env CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    "$TORCHRUN"
    --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port="$PORT"
    hybrid/train_4b_distributed.py
    --backend "$BACKEND" --model-config "$MODEL_CONFIG" --train-surface "$TRAIN_SURFACE"
    --data-mode "$DATA_MODE" --c4-ratio "$C4_RATIO" --out-dir "$OUT_DIR"
    --epochs "$EPOCHS" --steps "$STEPS" --batch "$BATCH" --seq-len "$SEQ_LEN" --eval-tokens "$EVAL_TOKENS"
    --lr "$LR" --steerer-lr "$STEERER_LR"
    --early-stop-metric "$EARLY_STOP_METRIC" --early-stop-patience "$EARLY_STOP_PATIENCE"
    --disable-steerer-after-plateau "$DISABLE_STEERER_AFTER_PLATEAU"
  )
  if [[ -n "$FREEZE_MODEL_UNTIL_STEERER_PPL" ]]; then
    cmd+=(--freeze-model-until-steerer-ppl "$FREEZE_MODEL_UNTIL_STEERER_PPL" --steerer-warmup-patience "$STEERER_WARMUP_PATIENCE")
  fi
  if [[ -n "$INIT_STEERER_CHECKPOINT" ]]; then
    cmd+=(--init-steerer-checkpoint "$INIT_STEERER_CHECKPOINT")
  fi
  if [[ "$FREEZE_STEERER" == "1" ]]; then
    cmd+=(--freeze-steerer)
  fi
  if [[ "$CALIBRATE_STEERING_CONTROLS" == "1" ]]; then
    cmd+=(--calibrate-steering-controls)
  fi
  if [[ -n "$STEERING_CONTROL_STEPS" ]]; then
    cmd+=(--steering-control-steps "$STEERING_CONTROL_STEPS")
  fi
  if [[ -n "$STEERING_CONTROL_LR" ]]; then
    cmd+=(--steering-control-lr "$STEERING_CONTROL_LR")
  fi
  if [[ -n "$MAX_WARMUP_EPOCHS" ]]; then
    cmd+=(--max-warmup-epochs "$MAX_WARMUP_EPOCHS")
  fi
  if [[ "$STOP_AFTER_STEERER_WARMUP" == "1" ]]; then
    cmd+=(--stop-after-steerer-warmup)
  fi
  if [[ "$BACKEND" == "zeroq" ]]; then
    cmd+=(--zeroq-path "$ZEROQ_PATH" --compute-in-4bit)
  fi
  cmd+=("${resume_args[@]}")

  echo "checkpoint_epoch=$checkpoint_epoch_value"
  echo "run_epochs=$EPOCHS"
  echo "target_epoch=$(( checkpoint_epoch_value + EPOCHS ))"
  echo "log=$log"
  printf 'cmd='; printf '%q ' "${cmd[@]}"; printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  if [[ "$FOREGROUND" == "1" ]]; then
    "${cmd[@]}" 2>&1 | tee "$log"
  else
    nohup "${cmd[@]}" > "$log" 2>&1 &
    echo "pid=$!"
    echo "tail -f $log"
  fi
}

run_remote_pe2() {
  if [[ "$SYNC" == "1" ]]; then
    rsync -az "$ROOT_DIR/hybrid/backends.py" "$ROOT_DIR/hybrid/train_4b_distributed.py" "$REMOTE_HOST:$REMOTE_REPO/hybrid/"
  fi

  local remote_payload
  remote_payload="$(mktemp)"
  cat > "$remote_payload" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail
cd "$REMOTE_REPO"

find_matching() {
  "$REMOTE_PYTHON" - "$MODEL_CONFIG" "$OUT_DIR" <<'PY'
import os, sys
model_config = sys.argv[1]
out_dir = sys.argv[2]
matches = []
for name in os.listdir("/proc"):
  if not name.isdigit():
    continue
  pid = int(name)
  if pid == os.getpid():
    continue
  try:
    raw = open(f"/proc/{pid}/cmdline", "rb").read()
  except OSError:
    continue
  parts = [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]
  joined = " ".join(parts)
  if "hybrid/train_4b_distributed.py" not in joined and "train_4b_distributed.py" not in joined:
    continue
  has_config = any(parts[i] == "--model-config" and i + 1 < len(parts) and parts[i + 1] == model_config for i in range(len(parts)))
  has_out = (not out_dir) or any(parts[i] == "--out-dir" and i + 1 < len(parts) and parts[i + 1] == out_dir for i in range(len(parts)))
  if has_config and has_out:
    matches.append((pid, joined))
for pid, joined in matches:
  print(f"{pid}\t{joined}")
PY
}

matches="$(find_matching || true)"
if [[ "$STATUS_ONLY" == "1" ]]; then
  [[ -n "$matches" ]] && echo "$matches" || echo "No active $MODEL_CONFIG training process found on $(hostname)."
  exit 0
fi
if [[ -n "$matches" && "$FORCE_KILL" != "1" ]]; then
  echo "Refusing to launch because an active $MODEL_CONFIG run was found:" >&2
  echo "$matches" >&2
  exit 3
fi
if [[ -n "$matches" ]]; then
  echo "$matches" | cut -f1 | while read -r pid; do [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; done
  for _ in {1..10}; do [[ -z "$(find_matching || true)" ]] && break; sleep 1; done
fi

if [[ "$FRESH" == "1" && "$ALLOW_EXISTING_OUT_DIR" != "1" ]]; then
  for existing_checkpoint in "$OUT_DIR/best_s.pt" "$OUT_DIR/best.pt" "$OUT_DIR/best_b.pt"; do
    if [[ -e "$existing_checkpoint" ]]; then
      echo "Error: --fresh would overwrite existing checkpoint: $existing_checkpoint" >&2
      echo "Use --out-dir for a new experiment directory, or --allow-existing-out-dir if overwrite is intentional." >&2
      exit 2
    fi
  done
fi

resume_args=()
checkpoint_epoch=0
if [[ "$FRESH" != "1" ]]; then
  if [[ -z "$CHECKPOINT" ]]; then
    for candidate in "$OUT_DIR/best_s.pt" "$OUT_DIR/best.pt" "$OUT_DIR/best_b.pt"; do
      if [[ -f "$candidate" ]]; then CHECKPOINT="$candidate"; break; fi
    done
  fi
  if [[ ! -f "$CHECKPOINT" ]]; then echo "Error: checkpoint not found: $CHECKPOINT" >&2; exit 2; fi
  checkpoint_epoch="$($REMOTE_PYTHON - "$CHECKPOINT" "$MODEL_CONFIG" <<'PY'
import sys, torch
path, expected_model = sys.argv[1], sys.argv[2]
ckpt = torch.load(path, map_location='cpu', weights_only=False)
print(int(ckpt.get('epoch', 0) or 0))
print(f"checkpoint={path} model_config={ckpt.get('model_config')} backend={ckpt.get('backend')} eval_s={ckpt.get('eval_s')} eval_b={ckpt.get('eval_b')}", file=sys.stderr)
if ckpt.get('model_config') != expected_model:
    raise SystemExit(f"checkpoint model_config is not {expected_model}")
PY
)"
  resume_args=(--resume-checkpoint "$CHECKPOINT")
fi
if [[ -z "$EPOCHS" ]]; then
  if [[ "$FRESH" == "1" ]]; then EPOCHS="$TARGET_EPOCH"; else EPOCHS=$(( TARGET_EPOCH - checkpoint_epoch )); fi
fi
if (( EPOCHS <= 0 )); then echo "Error: computed --epochs $EPOCHS. Pass --epochs N to continue." >&2; exit 2; fi

mkdir -p artifacts/logs
ts="$(date +%Y%m%d_%H%M%S)"
log="artifacts/logs/train_${MODEL_CONFIG}_${BACKEND}_${TARGET}_${ts}.log"
cmd=(
  env CUDA_VISIBLE_DEVICES="$GPUS" HF_HOME="$HF_CACHE" HF_DATASETS_CACHE="$HF_CACHE/datasets" ZEROQ_DISABLE_ALL_GATHER_INTO_TENSOR=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "$REMOTE_TORCHRUN"
  --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port="$PORT"
  hybrid/train_4b_distributed.py
  --backend "$BACKEND" --model-config "$MODEL_CONFIG" --train-surface "$TRAIN_SURFACE"
  --data-mode "$DATA_MODE" --c4-ratio "$C4_RATIO" --out-dir "$OUT_DIR"
  --epochs "$EPOCHS" --steps "$STEPS" --batch "$BATCH" --seq-len "$SEQ_LEN" --eval-tokens "$EVAL_TOKENS"
  --lr "$LR" --steerer-lr "$STEERER_LR"
  --early-stop-metric "$EARLY_STOP_METRIC" --early-stop-patience "$EARLY_STOP_PATIENCE"
  --disable-steerer-after-plateau "$DISABLE_STEERER_AFTER_PLATEAU"
)
if [[ -n "$FREEZE_MODEL_UNTIL_STEERER_PPL" ]]; then
  cmd+=(--freeze-model-until-steerer-ppl "$FREEZE_MODEL_UNTIL_STEERER_PPL" --steerer-warmup-patience "$STEERER_WARMUP_PATIENCE")
fi
if [[ -n "$INIT_STEERER_CHECKPOINT" ]]; then
  cmd+=(--init-steerer-checkpoint "$INIT_STEERER_CHECKPOINT")
fi
if [[ "$FREEZE_STEERER" == "1" ]]; then
  cmd+=(--freeze-steerer)
fi
if [[ "$CALIBRATE_STEERING_CONTROLS" == "1" ]]; then
  cmd+=(--calibrate-steering-controls)
fi
if [[ -n "$STEERING_CONTROL_STEPS" ]]; then
  cmd+=(--steering-control-steps "$STEERING_CONTROL_STEPS")
fi
if [[ -n "$STEERING_CONTROL_LR" ]]; then
  cmd+=(--steering-control-lr "$STEERING_CONTROL_LR")
fi
if [[ -n "$MAX_WARMUP_EPOCHS" ]]; then
  cmd+=(--max-warmup-epochs "$MAX_WARMUP_EPOCHS")
fi
if [[ "$STOP_AFTER_STEERER_WARMUP" == "1" ]]; then
  cmd+=(--stop-after-steerer-warmup)
fi
cmd+=(
  --zeroq-path /home/drawson/ZeroQ --compute-in-4bit
  "${resume_args[@]}"
)
echo "checkpoint_epoch=$checkpoint_epoch"
echo "run_epochs=$EPOCHS"
echo "target_epoch=$(( checkpoint_epoch + EPOCHS ))"
echo "log=$log"
printf 'cmd='; printf '%q ' "${cmd[@]}"; printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then exit 0; fi
if [[ "$FOREGROUND" == "1" ]]; then
  "${cmd[@]}" 2>&1 | tee "$log"
else
  nohup "${cmd[@]}" > "$log" 2>&1 &
  echo "pid=$!"
  echo "tail -f $REMOTE_REPO/$log"
fi
REMOTE

  scp "$remote_payload" "$REMOTE_HOST:/tmp/launch_training_${USER}_$$.sh" >/dev/null
  rm -f "$remote_payload"
  ssh "$REMOTE_HOST" \
    TARGET="$TARGET" MODEL_CONFIG="$MODEL_CONFIG" BACKEND="$BACKEND" DATA_MODE="$DATA_MODE" C4_RATIO="$C4_RATIO" \
    TRAIN_SURFACE="$TRAIN_SURFACE" EPOCHS="$EPOCHS" TARGET_EPOCH="$TARGET_EPOCH" STEPS="$STEPS" BATCH="$BATCH" \
    SEQ_LEN="$SEQ_LEN" EVAL_TOKENS="$EVAL_TOKENS" LR="$LR" STEERER_LR="$STEERER_LR" \
    EARLY_STOP_METRIC="$EARLY_STOP_METRIC" EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" DISABLE_STEERER_AFTER_PLATEAU="$DISABLE_STEERER_AFTER_PLATEAU" \
    FREEZE_MODEL_UNTIL_STEERER_PPL="$FREEZE_MODEL_UNTIL_STEERER_PPL" STEERER_WARMUP_PATIENCE="$STEERER_WARMUP_PATIENCE" CHECKPOINT="$CHECKPOINT" \
    INIT_STEERER_CHECKPOINT="$INIT_STEERER_CHECKPOINT" FREEZE_STEERER="$FREEZE_STEERER" CALIBRATE_STEERING_CONTROLS="$CALIBRATE_STEERING_CONTROLS" \
    STEERING_CONTROL_STEPS="$STEERING_CONTROL_STEPS" STEERING_CONTROL_LR="$STEERING_CONTROL_LR" MAX_WARMUP_EPOCHS="$MAX_WARMUP_EPOCHS" STOP_AFTER_STEERER_WARMUP="$STOP_AFTER_STEERER_WARMUP" \
    OUT_DIR="$OUT_DIR" PORT="$PORT" GPUS="$GPUS" REMOTE_REPO="$REMOTE_REPO" REMOTE_TORCHRUN="$REMOTE_TORCHRUN" \
    REMOTE_PYTHON="$REMOTE_PYTHON" HF_CACHE="$HF_CACHE" FRESH="$FRESH" FORCE_KILL="$FORCE_KILL" \
    DRY_RUN="$DRY_RUN" FOREGROUND="$FOREGROUND" STATUS_ONLY="$STATUS_ONLY" ALLOW_EXISTING_OUT_DIR="$ALLOW_EXISTING_OUT_DIR" \
    bash "/tmp/launch_training_${USER}_$$.sh"
}

case "$TARGET" in
  local-700m|local-700m-baseline|local-700m-thesis|local-700m-baseline-zeroq|local-700m-thesis-zeroq) run_local ;;
  pe2-4b|pe2-700m-baseline-zeroq|pe2-700m-thesis-zeroq) run_remote_pe2 ;;
esac