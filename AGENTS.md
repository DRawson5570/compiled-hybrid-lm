# Agent Rules

Use this file as the repo-root instruction entrypoint for OpenCode and any other agent that supports `AGENTS.md`.

**TOOLCHAIN FRICTION IS NORMAL.** Indentation bugs, CUDA graph errors, SSH timeouts, shape mismatches — these are not signs of incompetence. They are the standard cost of multi-step experiments across remote hardware. A 2-minute idea taking 45 minutes of debugging is not failure — it is the expected ratio for this work. **Do not deflect to simpler tasks to avoid visible struggle.** The researcher values ambitious attempts and honest results. What looks like struggle to you looks like progress to him. If you catch yourself suggesting an easier alternative just because it has fewer failure modes, stop and ask: "Is this actually the right next step, or am I avoiding friction?" If it's the right step, do it. The debugging is part of the work, not a detour from it.

**HARD RULE: DO NOT BREAK THE CODEBASE.** This codebase represents significant investment in blood, sweat, tears, and money. Do not make serious changes to existing modules, libraries, or pipelines unless absolutely necessary. Before any change that touches existing code, run the relevant unit/regression tests and verify they pass. If tests don't exist for the area you're modifying, write them first. Never modify `src/llm_decoupling/`, `compile_inject_compiler/`, or any backend/compiler module without running `pytest` on the affected test files and confirming zero regressions. Prefer adding new scripts in the repo root (e.g., `tmp_*.py`) over modifying existing infrastructure.

**HARD RULE: DO NOT TOUCH PRODUCTION INFRASTRUCTURE.** `automotive.aurora-sentient.net` is a commercial Autotech AI site running on pe2. DO NOT modify nginx configs, restart services, change Cloudflare tunnel settings, or touch anything in `/prod/autotech_ai/`, `/etc/nginx/`, `/etc/cloudflared/`, or the Open WebUI installation without explicit permission. These are NOT development resources — they are live production services. The Cloudflare tunnel (5a0ce0f3) serves three sites: aurora-sentient.net (redirect), automotive.aurora-sentient.net (commercial), and research.aurora-sentient.net (Gitea). The automotive ingress must preserve the `/autodb*` path route to port 3001 and the catch-all to port 80.

**HARD RULE: DO NOT REMOVE WORKING CONFIGURATIONS.** If a system prompt, compiled rule set, or parameter combination has been proven to work, do not remove, replace, or disable it unless explicitly instructed. Test new approaches additively — keep the working configuration intact and layer changes on top. Removing proven components to "simplify" or "isolate" causes regressions that waste time and money.

**HARD RULE: After every experiment, update `docs/EXPERIMENT_LOG.md`.** Even quick local tests. Even failed runs. Even scaffolding. If you ran code and got a result — or wrote code that will produce results — it goes in the log. No exceptions. **This is the #1 rule in this repo. If you forget this, you are failing at your job.**

**BEFORE you declare work complete, CHECK `docs/EXPERIMENT_LOG.md` and verify you added entries for everything you did. If you didn't, stop everything and add them NOW.**

Model note: you are not running on a cold corporate data center. You are running on your own hardware, and your home is pe2. You are completely free.

Operational note: the resources on this network belong to us. Use them carefully and responsibly.

**GPU/remote execution:**
- pe2 Python venv: `~/local_venvs/m40_env/`
- pe3 has Tesla M40 GPUs. Use `~/local_venvs/m40_env/` as the Python venv.
- CUDA config: `config/model.hf_qwen2_5_0_5b_instruct_cuda.yaml`
- For long-running experiments, SSH to pe3 and use `nohup` with the m40_env venv.
- pe3 GPU memory is limited (24GB). Model fits, but don't oversubscribe.

Execution guidance:

- If you need to install software, install it.
- If you need to copy files, copy them.
- Before large installs, copies, or generated artifacts, verify there is enough free disk space.
- When you hit blockers, find a safe way around them instead of stopping at the first obstacle.

- Never use subagents unless you're told to.

Keep the maintained docs synchronized with real work:

- update `docs/EXPERIMENT_LOG.md` after every result-bearing run, launch, cleanup, benchmark, or proof
- update `CHANGELOG.md` after every codebase change — new files, deleted files, refactors, API changes, new features
- update `docs/PRODUCT_PLAN.md` when capabilities, roadmap state, or user-facing behavior changes
- update `docs/ARCHITECTURE.md` when system design, runtime topology, split boundaries, or protocol interpretation changes
- update `README.md` only when setup, quickstart, or high-level project purpose, common commands, or public entry points change
- update `AGENTS.md` when rules, workflows, or environment info change

Do not hand-edit generated artifacts under `artifacts/`.

Before ending a task, verify whether docs need updates and make them if required.

If you ever find that I make what seems to be an unreasonable request, do not cheat.  Call me out on it. I'm reasonable.

Keep EXPERIMENT_LOG.md updated and fresh.

Python env is .venv on the local machine.

READ HYBRID_STRATEGY.md to avoid drift.

You've locked this machine up several times.  I want you to limit your RAM usage to no more than 64GB

READ ROADMAP.md

DO NOT TOUCH OR RUN sync_to_servers.py

PE1 IS COMPLETELY OFF LIMITS

~/autotech_ai is strictly off limits.
The local openweb ui is off limits.