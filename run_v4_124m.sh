#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# V4 124M steerer training — start fresh or resume.
# Original produced eval_b=35.6, eval_s=28.2 on WikiText-103.

EPOCHS="${EPOCHS:-200}"
STEPS="${STEPS:-500}"
BATCH="${BATCH:-8}"
SEQ_LEN="${SEQ_LEN:-128}"
LR="${LR:-1e-3}"
EVAL_TOKENS="${EVAL_TOKENS:-8192}"
OUT_DIR="${OUT_DIR:-artifacts/steerer_v4_replication}"
DEVICE="${DEVICE:-cuda}"
RESUME_CKPT="${RESUME_CKPT:-}"
FRESH=0

usage() {
  cat <<'USAGE'
Usage: ./run_v4_124m.sh [options]

  --fresh              Start from scratch (no V2 warm-start).
  --epochs N           Epochs to run. Default: 200
  --steps N            Steps per epoch. Default: 500
  --batch N            Batch size. Default: 8
  --seq-len N          Sequence length. Default: 128
  --lr X               Learning rate. Default: 1e-3
  --eval-tokens N      Validation tokens. Default: 8192
  --out-dir PATH       Output directory. Default: artifacts/steerer_v4_replication
  --device DEV         torch device. Default: cuda
  --resume-ckpt PATH   Resume from a train_steerer_v4 checkpoint.
  --kill               Kill existing train_steerer_v4 before launching.
  --dry-run            Print command without launching.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh) FRESH=1; shift ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --eval-tokens) EVAL_TOKENS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --resume-ckpt) RESUME_CKPT="$2"; shift 2 ;;
    --kill) kill $(pgrep -f train_steerer_v4) 2>/dev/null; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage >&2; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG="artifacts/logs/train_v4_124m_${TS}.log"
mkdir -p artifacts/logs

if [[ "$FRESH" == "1" ]]; then
  MODEL_ARG="--from-scratch --model-config 124m"
  ACTION="fresh from-scratch"
else
  MODEL_ARG="--resume-model artifacts/steerer_v2/steerer_best_b.pt"
  ACTION="warm-start from V2"
fi

RESUME_ARG=""
if [[ -n "$RESUME_CKPT" ]]; then
  RESUME_ARG="--resume-training-ckpt $RESUME_CKPT"
  ACTION="$ACTION + resume from $RESUME_CKPT"
fi

cmd=(.venv/bin/python hybrid/train_steerer_v4.py
  $MODEL_ARG
  --epochs "$EPOCHS"
  --steps "$STEPS"
  --batch "$BATCH"
  --seq-len "$SEQ_LEN"
  --lr "$LR"
  --out-dir "$OUT_DIR"
  --device "$DEVICE"
  --backend dense
  --eval-tokens "$EVAL_TOKENS"
  $RESUME_ARG
)

echo "action=$ACTION"
echo "out_dir=$OUT_DIR"
echo "log=$LOG"
printf 'cmd: %s\n' "${cmd[*]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

nohup "${cmd[@]}" > "$LOG" 2>&1 &
echo "pid=$!"
echo "tail -f $LOG"
