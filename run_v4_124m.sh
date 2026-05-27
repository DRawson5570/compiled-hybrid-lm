#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Replicate the V4 124M experiment that produced eval_s=28.2, eval_b=35.6.
# Same architecture, same data, same hyperparameters.
# Fresh start: no warm-start from V2 checkpoint.

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  .venv/bin/python hybrid/train_steerer_v4.py \
  --from-scratch \
  --model-config 124m \
  --epochs 200 \
  --steps 500 \
  --batch 8 \
  --seq-len 128 \
  --lr 1e-3 \
  --out-dir artifacts/steerer_v4_replication \
  --device cuda \
  --backend dense \
  --eval-tokens 8192 \
  --log-every 50
