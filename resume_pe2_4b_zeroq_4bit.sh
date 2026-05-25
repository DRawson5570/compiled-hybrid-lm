#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PE2_REPO="${PE2_REPO:-/home/drawson/deepseek_experiments}"
PE2_PYTHON="${PE2_PYTHON:-/home/drawson/local_venvs/m40_env/bin/python}"
PE2_TORCHRUN="${PE2_TORCHRUN:-/home/drawson/local_venvs/m40_env/bin/torchrun}"
CHECKPOINT="${CHECKPOINT:-artifacts/train_4b_cmi_steerer_zeroq_4bit/best.pt}"
EPOCHS="${EPOCHS:-}"
STEPS="${STEPS:-50}"
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-64}"
EVAL_TOKENS="${EVAL_TOKENS:-512}"
LR="${LR:-1e-4}"
STEERER_LR="${STEERER_LR:-1e-4}"
PORT="${PORT:-29567}"
GPUS="${GPUS:-0,1}"
FOREGROUND=0
SYNC=1

usage() {
  cat <<'USAGE'
Usage: ./resume_pe2_4b_zeroq_4bit.sh --epochs N [options]

Options:
  --epochs N          Additional epochs to run from the current checkpoint.
  --steps N           Steps per epoch. Default: 50
  --batch N           Batch size. Default: 1
  --seq-len N         Sequence length. Default: 64
  --eval-tokens N     Eval tokens. Default: 512
  --lr X              Model-surface learning rate. Default: 1e-4
  --steerer-lr X      Steerer learning rate. Default: 1e-4
  --checkpoint PATH   Checkpoint on pe2 to resume. Default: artifacts/train_4b_cmi_steerer_zeroq_4bit/best.pt
  --port N            torchrun master port. Default: 29567
  --gpus LIST         CUDA_VISIBLE_DEVICES list. Default: 0,1
  --no-sync           Do not rsync trainer/backend files before launch.
  --foreground        Run attached instead of detached nohup.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs) EPOCHS="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --eval-tokens) EVAL_TOKENS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --steerer-lr) STEERER_LR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --no-sync) SYNC=0; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$EPOCHS" ]]; then
  echo "Error: --epochs N is required." >&2
  usage >&2
  exit 2
fi

if [[ "$SYNC" == "1" ]]; then
  rsync -az \
    "$ROOT_DIR/hybrid/backends.py" \
    "$ROOT_DIR/hybrid/train_4b_distributed.py" \
    pe2:"$PE2_REPO/hybrid/"
fi

REMOTE_CMD=$(cat <<REMOTE
set -euo pipefail
cd "$PE2_REPO"
test -f "$CHECKPOINT"
mkdir -p artifacts/logs
ts=\$(date +%Y%m%d_%H%M%S)
log="artifacts/logs/resume_4b_cmi_steerer_zeroq_4bit_\${ts}.log"
cmd=(env CUDA_VISIBLE_DEVICES="$GPUS" ZEROQ_DISABLE_ALL_GATHER_INTO_TENSOR=1 "$PE2_TORCHRUN"
  --nproc_per_node=2
  --nnodes=1
  --node_rank=0
  --master_addr=localhost
  --master_port="$PORT"
  hybrid/train_4b_distributed.py
  --backend zeroq
  --model-config 4b
  --train-surface cmi_steerer
  --epochs "$EPOCHS"
  --steps "$STEPS"
  --batch "$BATCH"
  --seq-len "$SEQ_LEN"
  --eval-tokens "$EVAL_TOKENS"
  --lr "$LR"
  --steerer-lr "$STEERER_LR"
  --zeroq-path /home/drawson/ZeroQ
  --compute-in-4bit
  --resume-checkpoint "$CHECKPOINT")
echo "CHECKPOINT=$CHECKPOINT"
echo "LOG=\$log"
echo "CMD=\${cmd[*]}"
if [[ "$FOREGROUND" == "1" ]]; then
  "\${cmd[@]}" 2>&1 | tee "\$log"
else
  nohup "\${cmd[@]}" > "\$log" 2>&1 &
  echo "PID=\$!"
fi
REMOTE
)

ssh pe2 "$REMOTE_CMD"