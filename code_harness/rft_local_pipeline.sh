#!/bin/bash
# Local 3080 RFT pipeline: wait for candidate cache -> smoke RFT -> full RFT -> A/B eval.
# Base loads in 4-bit NF4 (~3.1GB), leaving ample headroom on the 10GB 3080.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/drawson/deepseek_experiments/.venv/bin/python
CH=/home/drawson/code_harness
ART=/home/drawson/deepseek_experiments/artifacts
CACHE=$CH/cand_mbpp_test.jsonl
LOG=/tmp/rft_local.log
exec >>"$LOG" 2>&1
echo "=== local RFT pipeline start $(date) ==="

# 1. wait for the candidate cache (gen writes it only at the very end)
while [ ! -s "$CACHE" ]; do echo "[$(date +%H:%M:%S)] waiting for cache..."; sleep 60; done
echo "[$(date +%H:%M:%S)] cache ready: $(wc -l < "$CACHE") problems"
# give the GPU a moment to fully release after gen exits
sleep 10

# 2. smoke test (catch runtime / OOM errors fast)
echo "=== SMOKE $(date) ==="
$PY $CH/train_rft.py --device cuda:0 --steps 30 --eval-every 15 --eval-n 8 \
   --eval-batch 1 --seq-cap 512 --out $ART/qwen35_4b_rft_smoke
RC=$?
echo "[$(date +%H:%M:%S)] smoke rc=$RC"
if [ $RC -ne 0 ]; then echo "SMOKE FAILED - aborting"; exit 1; fi

# 3. full run
echo "=== FULL RFT $(date) ==="
$PY $CH/train_rft.py --device cuda:0 --steps 1500 --eval-every 250 --eval-n 50 \
   --eval-batch 1 --seq-cap 512 --out $ART/qwen35_4b_rft
RC=$?
echo "[$(date +%H:%M:%S)] full rft rc=$RC"
if [ $RC -ne 0 ]; then echo "FULL RFT FAILED"; exit 1; fi

# 4. definitive A/B on full HumanEval
echo "=== A/B EVAL (full HumanEval) $(date) ==="
$PY $CH/eval_ab.py --ckpt $ART/qwen35_4b_rft/cartridge_best.pt --device cuda:0 --batch 1
echo "=== local RFT pipeline done $(date) ==="
