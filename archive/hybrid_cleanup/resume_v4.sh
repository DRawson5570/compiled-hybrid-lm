#!/bin/bash
# resume_v4.sh — Resume V4 steerer co-training from best checkpoint.
# Usage: ./resume_v4.sh --resume-epochs 100

set -e

EPOCHS=""
CHECKPOINT="artifacts/steerer_v4/steerer_best_b.pt"
NEURAL_CKPT="artifacts/c4_v2_768_x30/best.pt"
LOG="/tmp/steerer_v4_cont.log"

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume-epochs) EPOCHS="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [[ -z "$EPOCHS" ]]; then
    echo "Usage: $0 --resume-epochs N [--checkpoint PATH]"
    exit 1
fi

echo "=== Resume V4: +${EPOCHS} epochs | $(date) ===" | tee -a "$LOG"
echo "  Checkpoint: $CHECKPOINT" | tee -a "$LOG"
echo "  Log: $LOG" | tee -a "$LOG"

python3 -u train_steerer_v4.py \
    --neural-ckpt "$NEURAL_CKPT" \
    --resume-model "$CHECKPOINT" \
    --epochs "$EPOCHS" \
    --steps 500 \
    --batch 8 \
    >> "$LOG" 2>&1

echo "=== Done: $(date) ===" | tee -a "$LOG"
