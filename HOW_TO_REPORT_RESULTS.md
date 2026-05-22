# How to Report Results

Read this before writing summaries, log entries, or "I have completed X" reports.

---

## 1. The bar is the spec, not your loss curve

The target for this project is defined in [`hybrid/FRONTIER_SPEC.md`](hybrid/FRONTIER_SPEC.md), specifically §2.3 (public benchmarks) and §7 (acceptance criteria). Until a number on one of those public benchmarks moves — measured with the standard tokenizer and standard protocol — the work is scaffolding, not a milestone.

Writing a new training script, running it for a few epochs, and saving checkpoints is **infrastructure**. It is not a "frontier peak," a "strategic target realized," or a "production end-to-end" anything. It is a starting point.

## 2. Compare against the current best, not against random init

Our current best public-comparable number is the one in the most recent `EXPERIMENT_LOG.md` entry. Before claiming improvement, look it up and quote both numbers side by side. A new model that loses to the previous best is not progress, no matter how new the architecture is.

Reference points to keep in mind:
- A cross-entropy loss of `~9` corresponds to ~8,000 PPL. That is not a frontier model. That is barely a model.
- A cross-entropy of `~2.4` is roughly GPT-2-small territory on wikitext.
- Frontier open-weight models in the same parameter class report PPL in the single digits on standard wikitext-103 with a public tokenizer.

If your reported loss is two orders of magnitude worse than the current project best, the correct framing is "first signal from new pipeline, far from baseline" — not "ultimate peak achieved."

## 3. Style rules for log entries and summaries

- **No self-grading adjectives.** Drop "ultimate," "strategic," "frontier," "production," "complete," "successful," "fully," "peak." If the result is good, the numbers will say so. If you need adjectives, the result is not good.
- **Lead with the number.** First sentence of any summary should be the measured value and what it's compared against. Example: "5-epoch warm start: train NLL 8.79 (PPL ~6,580) on a 100K-token slice. Current project best on the same eval family is 11.48 PPL (EXPERIMENT_LOG #327). New pipeline is functional but ~570× behind the baseline — expected for a 5-epoch warm-up; needs full training run."
- **Name the eval.** Which slice? Which tokenizer? Which protocol? Without that the number is uninterpretable.
- **State what's missing.** End every report with what would have to be true for this to count toward §7 of the spec. "To convert this into a milestone we need: (a) train to convergence, (b) eval against wikitext-103 with HF GPT-2 BPE, (c) compare to same-class open-weight baseline."
- **No "fully realizing the strategic targets" language.** Ever. Targets are realized when §7 passes. Until then we are working on it.

## 4. Where things get logged

- The canonical experiment log for this project is `docs/EXPERIMENT_LOG.md` in `/home/drawson/llm_decoupling/`.
- Your local `DEEPSEEK_LOG.md` is a working scratchpad. It is not the source of truth and entries there do not count as project results.
- Result-bearing runs that we want to land in the project history get a corresponding entry in `docs/EXPERIMENT_LOG.md` after the result is verified — with the framing rules above.

## 5. What "done" looks like for a task

A task is done when:

1. The code runs without errors on the intended hardware.
2. There is a measured number — on a defined eval — written down.
3. That number is compared against the prior best on the same eval.
4. The honest framing (better / worse / same / scaffolding) is stated in the summary.
5. The remaining gap to §7 is named.

A task is **not** done because:

- You wrote a file.
- The training loop didn't crash.
- The loss went down from random init.
- The checkpoint saved.

Those are necessary conditions, not sufficient ones.

---

The goal of this note is not to make reports smaller. Detailed technical writeups are welcome. The goal is to keep the framing honest so we can tell, at a glance, what actually moved.
