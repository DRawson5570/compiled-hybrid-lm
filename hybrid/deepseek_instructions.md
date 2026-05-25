# DeepSeek Instructions: ZeroQ CMI Backend Test

Goal: prove the new CMI Hybrid backend path works before launching another large pe3 run.

## Current Branch State

- Work is merged into `main` at `24ba348 Add ZeroQ backend substrate`.
- ZeroQ is now an optional backend under the cartridge ABI, not part of the cartridge API itself.
- Key files:
  - `hybrid/backends.py` — `DenseTorchBackend`, `ZeroQPartitionedBackend`, `TrainableSurface`.
  - `hybrid/hf_deepseek.py` — explicit-Linear HF-compatible backbone for ZeroQ and cartridge hooks.
  - `hybrid/train_4b_distributed.py` — distributed trainer with `--backend dense|zeroq`.
  - `hybrid/tests/test_backends.py` and `hybrid/tests/test_hf_deepseek.py` — local tests.

## Do Not Drift

- Do not switch this into pure full-model SGD. Keep the base frozen or mostly frozen.
- The large pe3/M40 smoke surface is `head_bias`; the `cmi_steerer` surface is available but OOMs on the 3B config until we add activation checkpointing or a custom frozen-backbone backward path.
- Compiled priors must be used as activation steering features, not just loaded and ignored.
- Do not kill the existing pe3 ZeroQ run unless Douglas explicitly asks.

## Local CPU/GPU Sanity Test

Run this first on the local workstation:

```bash
cd ~/deepseek_experiments
.venv/bin/python -m pytest hybrid/tests/test_backends.py hybrid/tests/test_hf_deepseek.py hybrid/tests/test_cartridges.py -q
.venv/bin/python -m py_compile hybrid/backends.py hybrid/hf_deepseek.py hybrid/train_4b_distributed.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --nproc_per_node=1 hybrid/train_4b_distributed.py \
  --backend dense \
  --model-config test \
  --train-surface cmi_steerer \
  --epochs 1 \
  --steps 1 \
  --batch 1 \
  --seq-len 16 \
  --eval-tokens 128 \
  --lr 1e-4 \
  --steerer-lr 1e-4
```

Expected result:

- Tests pass.
- Compile passes.
- Trainer prints `backend=dense config=test`.
- Trainer mounts 2 steerer hooks.
- Trainer reports `model_trainable=50,257` and a small steerer trainable count on the test config.
- Trainer completes one eval and saves under ignored `artifacts/train_3b/`.

## pe3 ZeroQ Smoke Test

Only run this when pe3 has a GPU slot free and no important run would be displaced:

```bash
ssh pe3 "cd ~/deepseek_experiments && \
  CUDA_VISIBLE_DEVICES=0,1 ~/local_venvs/m40_env/bin/torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29531 \
    hybrid/train_4b_distributed.py \
      --backend zeroq \
      --model-config test \
      --train-surface cmi_steerer \
      --epochs 1 \
      --steps 1 \
      --batch 1 \
      --seq-len 16 \
      --eval-tokens 128 \
      --zeroq-path ~/ZeroQ"
```

Expected result:

- Trainer prints `backend=zeroq config=test`.
- ZeroQ prepares the frozen backbone without full-model GPU OOM.
- Steerer hooks mount and compiled features are fed during the step.
- Both ranks complete the one-step run.

## Full 3B Test After Smoke Passes

For the real pe3 3B smoke test, use the memory-safe head-bias surface:

```bash
ssh pe3 "cd ~/deepseek_experiments && \
  CUDA_VISIBLE_DEVICES=0,1 ~/local_venvs/m40_env/bin/torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29545 \
    hybrid/train_4b_distributed.py \
      --backend zeroq \
      --model-config 3b \
      --train-surface head_bias \
      --epochs 3 \
      --steps 1 \
      --batch 1 \
      --seq-len 16 \
      --eval-tokens 128 \
      --lr 1e-4 \
      --zeroq-path ~/ZeroQ"
```

Known result from 2026-05-25: this completed 3 epochs on pe3, stayed around 1.5-1.8 GiB/GPU, and saved `artifacts/train_3b/best.pt` with metadata `train_surface=head_bias`, `epoch=3`, `eval_s=92816.46`.

## What To Report Back

Report these exact fields:

- commit SHA
- host and GPUs
- command
- peak GPU memory per rank
- `model_trainable` and `steerer_trainable`
- whether compiled-prior steerer hooks mounted
- final loss/eval line
- artifact path if a checkpoint was saved