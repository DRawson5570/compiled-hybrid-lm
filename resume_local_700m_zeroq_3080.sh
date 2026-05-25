#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${CHECKPOINT:-artifacts/train_700m_cmi_steerer_zeroq_4bit/best.pt}"
TARGET_EPOCH="${TARGET_EPOCH:-100}"
EPOCHS="${EPOCHS:-}"
STEPS="${STEPS:-240}"
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-64}"
EVAL_TOKENS="${EVAL_TOKENS:-512}"
LR="${LR:-1e-4}"
STEERER_LR="${STEERER_LR:-1e-4}"
PORT="${PORT:-29583}"
GPUS="${GPUS:-0}"
ZEROQ_PATH="${ZEROQ_PATH:-$HOME/ZeroQ}"
TORCHRUN="${TORCHRUN:-$ROOT_DIR/.venv/bin/torchrun}"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-artifacts/logs}"
FOREGROUND=0
FRESH=0
FORCE_KILL=0
DRY_RUN=0
STATUS_ONLY=0

usage() {
  cat <<'USAGE'
Usage: ./resume_local_700m_zeroq_3080.sh [options]

Resume or restart the local RTX 3080 700M ZeroQ CMI steerer run.

  --fresh               Start from scratch instead of loading a checkpoint.
  --epochs N            Additional epochs to run. Defaults to target - checkpoint epoch.
  --target-epoch N      Target absolute epoch when --epochs is omitted. Default: 100
  --steps N             Steps per epoch. Default: 240
  --batch N             Batch size. Default: 1
  --seq-len N           Sequence length. Default: 64
  --eval-tokens N       Eval tokens. Default: 512
  --lr X                Model-surface learning rate. Default: 1e-4
  --steerer-lr X        Steerer learning rate. Default: 1e-4
  --checkpoint PATH     Checkpoint to resume. Default: artifacts/train_700m_cmi_steerer_zeroq_4bit/best.pt
  --port N              torchrun master port. Default: 29583
  --gpus LIST           CUDA_VISIBLE_DEVICES list. Default: 0
  --zeroq-path PATH     ZeroQ checkout path. Default: $HOME/ZeroQ
  --torchrun PATH       torchrun executable. Default: .venv/bin/torchrun
  --python PATH         Python executable used for validation. Default: .venv/bin/python
  --log-dir PATH        Log directory. Default: artifacts/logs
  --status              Show active 700M train processes and exit.
  --force-kill          Kill active matching 700M train processes before launch.
  --dry-run             Print the command without launching.
  --foreground          Run attached instead of detached nohup.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh) FRESH=1; shift ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --target-epoch) TARGET_EPOCH="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --seq-len) SEQ_LEN="$2"; shift 2 ;;
    --eval-tokens) EVAL_TOKENS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --steerer-lr) STEERER_LR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --zeroq-path) ZEROQ_PATH="$2"; shift 2 ;;
    --torchrun) TORCHRUN="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --status) STATUS_ONLY=1; shift ;;
    --force-kill) FORCE_KILL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$TORCHRUN" ]]; then
  echo "Error: torchrun not executable: $TORCHRUN" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Error: python not executable: $PYTHON" >&2
  exit 2
fi

find_matching_pids() {
  "$PYTHON" - <<'PY'
import os
root = os.getcwd()
matches = []
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid == os.getpid():
        continue
    try:
        raw = open(f'/proc/{pid}/cmdline', 'rb').read()
    except OSError:
        continue
    parts = [p.decode('utf-8', 'replace') for p in raw.split(b'\0') if p]
    joined = ' '.join(parts)
    if 'hybrid/train_4b_distributed.py' not in joined and 'train_4b_distributed.py' not in joined:
        continue
    if '--model-config 700m' not in joined and not any(parts[i] == '--model-config' and i + 1 < len(parts) and parts[i + 1] == '700m' for i in range(len(parts))):
        continue
    matches.append((pid, joined))
for pid, joined in matches:
    print(f'{pid}\t{joined}')
PY
}

MATCHES="$(find_matching_pids || true)"
if [[ "$STATUS_ONLY" == "1" ]]; then
  if [[ -n "$MATCHES" ]]; then
    echo "$MATCHES"
  else
    echo "No active local 700M train_4b_distributed.py process found."
  fi
  exit 0
fi

if [[ -n "$MATCHES" && "$FORCE_KILL" != "1" ]]; then
  echo "Refusing to launch because an active local 700M run was found:" >&2
  echo "$MATCHES" >&2
  echo "Use --force-kill only when you intentionally want to stop and replace it." >&2
  exit 3
fi

if [[ -n "$MATCHES" && "$FORCE_KILL" == "1" ]]; then
  echo "$MATCHES" | cut -f1 | while read -r pid; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  for _ in {1..10}; do
    REMAINING="$(find_matching_pids || true)"
    [[ -z "$REMAINING" ]] && break
    sleep 1
  done
  REMAINING="$(find_matching_pids || true)"
  if [[ -n "$REMAINING" ]]; then
    echo "$REMAINING" | cut -f1 | while read -r pid; do
      [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
    done
  fi
  for _ in {1..5}; do
    REMAINING="$(find_matching_pids || true)"
    [[ -z "$REMAINING" ]] && break
    sleep 1
  done
  REMAINING="$(find_matching_pids || true)"
  if [[ -n "$REMAINING" ]]; then
    echo "Error: matching 700M processes still running after --force-kill:" >&2
    echo "$REMAINING" >&2
    exit 4
  fi
fi

RESUME_ARGS=()
CHECKPOINT_EPOCH=0
if [[ "$FRESH" != "1" ]]; then
  if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: checkpoint not found: $CHECKPOINT" >&2
    exit 2
  fi
  CHECKPOINT_EPOCH="$($PYTHON - "$CHECKPOINT" <<'PY'
import sys, torch
path = sys.argv[1]
ckpt = torch.load(path, map_location='cpu', weights_only=False)
print(int(ckpt.get('epoch', 0) or 0))
print(f"checkpoint={path} model_config={ckpt.get('model_config')} surface={ckpt.get('train_surface')} backend={ckpt.get('backend')} compute_in_4bit={ckpt.get('compute_in_4bit')} eval_s={ckpt.get('eval_s')} eval_b={ckpt.get('eval_b')}", file=sys.stderr)
if ckpt.get('model_config') != '700m':
    raise SystemExit('checkpoint model_config is not 700m')
if ckpt.get('backend') != 'zeroq':
    raise SystemExit('checkpoint backend is not zeroq')
if ckpt.get('compute_in_4bit') is not True:
    raise SystemExit('checkpoint was not trained with compute_in_4bit=True')
PY
)"
  RESUME_ARGS=(--resume-checkpoint "$CHECKPOINT")
fi

if [[ -z "$EPOCHS" ]]; then
  if [[ "$FRESH" == "1" ]]; then
    EPOCHS="$TARGET_EPOCH"
  else
    EPOCHS=$(( TARGET_EPOCH - CHECKPOINT_EPOCH ))
  fi
fi

if (( EPOCHS <= 0 )); then
  echo "Error: computed --epochs $EPOCHS from checkpoint epoch $CHECKPOINT_EPOCH and target $TARGET_EPOCH." >&2
  echo "Pass --epochs N to continue past the target." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"
log="$LOG_DIR/train_700m_zeroq_3080_resume_${ts}.log"

cmd=(
  env CUDA_VISIBLE_DEVICES="$GPUS" ZEROQ_DISABLE_ALL_GATHER_INTO_TENSOR=1 "$TORCHRUN"
  --nproc_per_node=1
  --nnodes=1
  --node_rank=0
  --master_addr=localhost
  --master_port="$PORT"
  hybrid/train_4b_distributed.py
  --backend zeroq
  --model-config 700m
  --train-surface cmi_steerer
  --epochs "$EPOCHS"
  --steps "$STEPS"
  --batch "$BATCH"
  --seq-len "$SEQ_LEN"
  --eval-tokens "$EVAL_TOKENS"
  --lr "$LR"
  --steerer-lr "$STEERER_LR"
  --zeroq-path "$ZEROQ_PATH"
  --compute-in-4bit
  "${RESUME_ARGS[@]}"
)

echo "checkpoint_epoch=$CHECKPOINT_EPOCH"
echo "run_epochs=$EPOCHS"
echo "target_epoch=$(( CHECKPOINT_EPOCH + EPOCHS ))"
echo "log=$log"
printf 'cmd='; printf '%q ' "${cmd[@]}"; printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ "$FOREGROUND" == "1" ]]; then
  "${cmd[@]}" 2>&1 | tee "$log"
else
  nohup "${cmd[@]}" > "$log" 2>&1 &
  pid="$!"
  echo "pid=$pid"
  echo "tail -f $log"
fi