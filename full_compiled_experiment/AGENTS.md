# Agent Rules

See [COMPILED_PRIOR_THESIS.md](COMPILED_PRIOR_THESIS.md) first, then [docs/HYBRID_STRATEGY.md](docs/HYBRID_STRATEGY.md).

## Before Declaring Something Impossible

1. **Check the git log.** Other agents (GPT-5.5, etc.) may have already fixed the problem.
   `git log --oneline --all -20` before spending hours on a dead end.

2. **fp16 numeric stability.** Any custom math in forward hooks (RMS norm, division, softmax of small values) must cast to float32 first, then cast back. fp16 range is ±65504 with min subnormal ~6e-8. `h.pow(2)` in fp16 overflows or underflows silently.

3. **Activation steering NaN checklist (SuperpositionSteererV2/V3 only):**
   - Is `init_scale` large enough that `o_rms` isn't zero? (Rule of thumb: `init_scale >= 0.5` for d_model >= 768)
   - Are the RMS calculations in float32? (fp16 math in hooks = NaN)
   - Is the model dtype consistent? (Steerer float32 + model float16 = cast output back)
   Note: FeatureConditionedAdapterSteerer uses `init_scale=0.005` and runs in fp32 — different checklist.

4. **When stuck on a training result**, sweep the obvious hyperparameter before changing architecture:
   - `init_scale`
   - `gamma`
   - `lr`
   - `noise_scale`

## Cartridge Training — Required Sanity Checks

*Added by Qwen3.6 at 2026-05-29 after a 5-hour dead end (EXPERIMENT_LOG #388).*

5. **Prove it on one problem first.** Before training on 295 examples for 2000 steps, prove the cartridge fixes ONE known-working problem in 50 steps. If loss drops but output doesn't change, stop — gradients aren't flowing.

6. **Verify gradient flow after gradient checkpointing.** `model.gradient_checkpointing_enable()` with a frozen base silently kills hook gradients unless `gradient_checkpointing_kwargs={"use_reentrant": False}`. After enabling GC, zero the steerer, run one train_step, and check that steerer params changed. A 3-line sanity check saves 5 hours.

7. **Autoregressive CE on canonical solutions is off-policy.** The model conditions on the gold prefix during training but on its own output during inference. Compounding errors defeat token-level accuracy. For open-ended generation tasks, train on the model's OWN rollouts (on-policy execution-feedback / RFT / STaR), not canonical solutions. Option-ranking CE works for multiple-choice; it does NOT transfer to generation.

## GPU-Specific

8. **M40 (Maxwell, compute 5.2):** No fp16 tensor cores. fp16 saves memory bandwidth only. Compute happens in fp32. Loading in fp16 then doing fp32 math = wasted conversions. Either use fp32 or profile carefully.

9. **bitsandbytes version lock:** M40 requires bitsandbytes <= 0.41.3. Newer versions (0.46.1+) dropped Maxwell support. HF's BitsAndBytesConfig requires >= 0.46.1 — use ZeroQ's functional quantize directly instead.

10. **pe2/pe3 GPU inventory:** pe3 has 2× Tesla M40 12GB (not 24GB — check `nvidia-smi`). pe2 has 5× Tesla M40 24GB.

## Collaboration

11. **Check for other agents' work first.** GPT-5.5 may have a feature branch with work in progress. Don't duplicate.
   `git branch -a` and `git log feature/* --oneline`

12. **When another agent fixes your bug**, read the diff carefully. The fix is usually a one-liner you missed.

13. **Document your dead ends.** Other agents need to know what you tried and why it failed. Use `FAILURE_ANALYSIS.md` or append to `EXPERIMENT_LOG.md`.

## File Rules

14. **Do not modify `hybrid/` files without explicit permission.** It's the public `compiled-hybrid-lm` repo.

15. **Work in the right folder:**
    - Hybrid model work → `~/deepseek_experiments/hybrid/`
    - ZeroQ → `~/ZeroQ/`
    - Code harness → `~/code_harness/`

16. **Treat the public repo with respect.** No debug prints, no ad-hoc scripts, no temp files, no agent artifacts committed to the public tree. No force-pushing through conflicts. No committing files that belong in local workspace directories. If it's not a core library file or project documentation, it goes in `archive/` locally — never in the repo. People link to this code from papers and social posts.

17. **Private-first benchmark and experiment work.** New benchmark lanes, exploratory prompts, failed scoring attempts, full per-question reports, local dataset caches, and generated artifacts should start in `~/deepseek_experiments`, not the public repo. Promote code back to `~/llm_decoupling` only after the harness is clean, non-leaky, reproducible, tested, and stripped of private paths/artifacts. For ARC specifically, use `~/deepseek_experiments/ARC_CHALLENGE_SPEC.md` as the implementation spec and update `EXPERIMENT_LOG.md` after real runs.
