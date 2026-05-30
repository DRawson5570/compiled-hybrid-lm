#!/bin/bash
# Launch joint MetaCompiler training on pe3 (M40 12GB GPU 0)
# Usage: ./scripts/launch_pe3_joint_train.sh

set -e

REMOTE="pe3"
REMOTE_DIR="~/deepseek_experiments/hybrid/full_compiled_experiment"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STEPS="${1:-2000}"
LR="${2:-1e-4}"

echo "Syncing code to $REMOTE..."
rsync -aH --delete \
  --exclude='artifacts/*' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

echo "Launching training on $REMOTE (steps=$STEPS, lr=$LR)..."
ssh "$REMOTE" "cd $REMOTE_DIR && \
  source ~/local_venvs/m40_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 \
  HF_HUB_OFFLINE=1 \
  PYTHONPATH=. \
  python3 scripts/train_joint_multilayer.py \
    --steps $STEPS \
    --lr $LR \
    --accum 4 \
    --device cuda \
    2>&1 | tee artifacts/joint_training/pe3_run_\$(date +%Y%m%d_%H%M%S).log"

echo "Done. Check results on pe3:"
echo "  ssh $REMOTE 'ls -la $REMOTE_DIR/artifacts/joint_training/'"
echo "  ssh $REMOTE 'cat $REMOTE_DIR/artifacts/joint_training/config.json'"
