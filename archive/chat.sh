#!/bin/bash
# chat.sh — Launch GPT-2 BPE blended chat with the C4-trained neural LM
# Usage: ./chat.sh [--alpha 0.5]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEEPSEEK="$(dirname "$SCRIPT_DIR")"
cd "$DEEPSEEK"

CKPT="$DEEPSEEK/artifacts/c4_v2_768_x30/best.pt"
BUILDER="$DEEPSEEK/artifacts/compiled_builder_50m.pt"
TRAIN_IDS="$DEEPSEEK/artifacts/wikitext_gpt2/train_ids.pt"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found at $CKPT"
    echo "Training still running? Check: tail -3 /tmp/c4_simple.log"
    exit 1
fi

echo "=== GPT-2 BPE Blended Chat ==="
echo "Checkpoint: $CKPT"
echo ""

python3 -u hybrid/generate_gpt2_blend.py \
    --ckpt "$CKPT" \
    --builder "$BUILDER" \
    --train-ids "$TRAIN_IDS" \
    --chat \
    "$@"
