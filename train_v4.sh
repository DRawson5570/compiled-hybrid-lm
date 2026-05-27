#!/usr/bin/env bash
# V4 steerer trainer — simple, no warmup gates. Local or remote.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="${MODEL:-124m}"
FRESH=0
EPOCHS="${EPOCHS:-}"
TARGET_EPOCH="${TARGET_EPOCH:-200}"
STEPS="${STEPS:-500}"
BATCH="${BATCH:-8}"
SEQ_LEN="${SEQ_LEN:-128}"
LR="${LR:-1e-3}"
EVAL_TOKENS="${EVAL_TOKENS:-8192}"
OUT_DIR="${OUT_DIR:-}"
BACKEND="${BACKEND:-dense}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-}"
ZEROQ_PATH="${ZEROQ_PATH:-$HOME/ZeroQ}"
COMPUTE_4BIT=0
REMOTE="${REMOTE:-}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/home/drawson/local_venvs/m40_env/bin/python3}"
REMOTE_TORCHRUN="${REMOTE_TORCHRUN:-/home/drawson/local_venvs/m40_env/bin/torchrun}"
REMOTE_REPO="${REMOTE_REPO:-/home/drawson/deepseek_experiments}"
HF_CACHE="${HF_CACHE:-}"
GPUS="${GPUS:-0}"
PORT="${PORT:-29567}"
FOREGROUND=0
DRY_RUN=0
KILL=0
FORCE_OUT=0

usage() { cat <<'USAGE'
Usage: ./train_v4.sh [options]

  --model CONFIG      124m, 500m, 1b, 4b. Default: 124m
  --fresh             Start from random weights.
  --epochs N          Epochs. Default: 200 (or target - checkpoint)
  --target-epoch N    Target absolute epoch. Default: 200
  --steps N           Steps/epoch. Default: 500
  --batch N           Batch size. Default: 8
  --seq-len N         Sequence length. Default: 128
  --lr X              Learning rate. Default: 1e-3
  --eval-tokens N     Validation tokens. Default: 8192
  --out-dir PATH      Output directory. Default: artifacts/steerer_v4_<MODEL>
  --backend MODE      dense or zeroq. Default: dense
  --compute-in-4bit   Enable 4-bit compute (ZeroQ only).
  --checkpoint PATH   Resume from this checkpoint (auto-detects from out-dir).
  --remote HOST       Launch on remote host (pe2, pe3).
  --gpus LIST         CUDA_VISIBLE_DEVICES. Default: 0
  --hf-cache PATH     HF datasets cache path for remote hosts.
  --foreground        Run attached (don't detach).
  --force-out-dir     Allow overwriting existing output directory.
  --kill              Kill matching processes before launch.
  --dry-run           Print command without launching.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --fresh) FRESH=1; shift ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --target-epoch) TARGET_EPOCH="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --eval-tokens) EVAL_TOKENS="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --compute-in-4bit) COMPUTE_4BIT=1; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --remote) REMOTE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --hf-cache) HF_CACHE="$2"; shift 2 ;;
    --foreground) FOREGROUND=1; shift ;;
    --force-out-dir) FORCE_OUT=1; shift ;;
    --kill) KILL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -z "$OUT_DIR" ]] && OUT_DIR="artifacts/steerer_v4_${MODEL}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="artifacts/logs/train_v4_${MODEL}_${TS}.log"
mkdir -p artifacts/logs

# Auto-detect checkpoint
if [[ "$FRESH" != "1" && -z "$CHECKPOINT" ]]; then
  for c in "$OUT_DIR/steerer_best_s.pt" "$OUT_DIR/steerer_best_b.pt" "$OUT_DIR/best.pt"; do
    [[ -f "$c" ]] && CHECKPOINT="$c" && break
  done
fi

# Out-dir safety
if [[ "$FORCE_OUT" != "1" && -d "$OUT_DIR" ]]; then
  for f in steerer_best_s.pt steerer_best_b.pt; do
    [[ -f "$OUT_DIR/$f" ]] && echo "Error: $OUT_DIR/$f exists. Use --force-out-dir." >&2 && exit 2
  done
fi

# Build command
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$ROOT/.venv/bin/torchrun}"

if [[ -n "$REMOTE" ]]; then
  PYTHON="$REMOTE_PYTHON"
  TORCHRUN="$REMOTE_TORCHRUN"
  WORKDIR="$REMOTE_REPO"
else
  WORKDIR="$ROOT"
fi

MODEL_ARG="--from-scratch --model-config $MODEL"
ACTION="fresh $MODEL"
if [[ "$MODEL" == "124m" && "$FRESH" != "1" ]]; then
  MODEL_ARG="--resume-model artifacts/steerer_v2/steerer_best_b.pt"
  ACTION="warm-start 124m from V2"
fi

# Resume from training checkpoint
RESUME_FLAG=""
CKPT_EPOCH=0
if [[ -n "$CHECKPOINT" ]]; then
  RESUME_FLAG="--resume-training-ckpt $CHECKPOINT"
  ACTION="$ACTION + resume"
  if [[ -f "$CHECKPOINT" ]]; then
    CKPT_EPOCH=$("$PYTHON" -c "import torch; c=torch.load('$CHECKPOINT',map_location='cpu',weights_only=False); print(int(c.get('epoch',0)or 0))" 2>/dev/null || echo 0)
  fi
fi

if [[ -z "$EPOCHS" ]]; then
  if [[ "$FRESH" == "1" || "$CKPT_EPOCH" == "0" ]]; then
    EPOCHS="$TARGET_EPOCH"
  else
    EPOCHS=$(( TARGET_EPOCH - CKPT_EPOCH ))
  fi
fi
(( EPOCHS <= 0 )) && echo "Error: target epoch $TARGET_EPOCH <= checkpoint $CKPT_EPOCH" >&2 && exit 2

ZEROQ_ARGS=""
[[ "$BACKEND" == "zeroq" ]] && ZEROQ_ARGS="--zeroq-path $ZEROQ_PATH"
[[ "$COMPUTE_4BIT" == "1" ]] && ZEROQ_ARGS="$ZEROQ_ARGS --compute-in-4bit"

ENV="env CUDA_VISIBLE_DEVICES=$GPUS"
[[ -n "$HF_CACHE" ]] && ENV="$ENV HF_DATASETS_CACHE=$HF_CACHE"

CMD=($ENV $TORCHRUN --nproc_per_node=1 --master_port=$PORT
  hybrid/train_steerer_v4.py
  $MODEL_ARG --epochs $EPOCHS --steps $STEPS --batch $BATCH --seq-len $SEQ_LEN
  --lr $LR --out-dir $OUT_DIR --device cuda --backend $BACKEND
  --eval-tokens $EVAL_TOKENS $ZEROQ_ARGS $RESUME_FLAG)

echo "action=$ACTION"
echo "checkpoint_epoch=$CKPT_EPOCH target_epoch=$((CKPT_EPOCH + EPOCHS))"
echo "out_dir=$OUT_DIR remote=$REMOTE backend=$BACKEND"
echo "log=$LOG"
printf 'cmd: %s\n' "${CMD[*]}"
[[ "$DRY_RUN" == "1" ]] && exit 0

if [[ -n "$REMOTE" ]]; then
  rsync -az hybrid/train_steerer_v4.py hybrid/train_scaled_neural_lm.py \
    hybrid/superposition_steerer_v3.py hybrid/gpu_channels.py hybrid/backends.py \
    "$REMOTE:$REMOTE_REPO/hybrid/" >/dev/null
  REMOTE_SCRIPT="/tmp/train_v4_$$.sh"
  printf '#!/bin/bash\nset -euo pipefail\ncd %s\n' "$REMOTE_REPO" > "/tmp/train_v4_remote_$$.sh"
  [[ "$KILL" == "1" ]] && printf 'pkill -f train_steerer_v4 || true\nsleep 2\n' >> "/tmp/train_v4_remote_$$.sh"
  printf '%s ' "${CMD[@]}" >> "/tmp/train_v4_remote_$$.sh"
  printf ' > %s 2>&1 &\necho "pid=$!"\n' "$LOG" >> "/tmp/train_v4_remote_$$.sh"
  scp "/tmp/train_v4_remote_$$.sh" "$REMOTE:$REMOTE_SCRIPT" >/dev/null
  ssh "$REMOTE" "bash $REMOTE_SCRIPT"
  rm -f "/tmp/train_v4_remote_$$.sh"
else
  [[ "$KILL" == "1" ]] && pkill -f 'train_steerer_v4.py' 2>/dev/null && sleep 2
  if [[ "$FOREGROUND" == "1" ]]; then
    "${CMD[@]}" 2>&1 | tee "$LOG"
  else
    nohup "${CMD[@]}" > "$LOG" 2>&1 &
    echo "pid=$!"
    echo "tail -f $LOG"
  fi
fi