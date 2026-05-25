#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

DEVICE="${DEVICE:-cuda}"
CHAT_CARTRIDGE="${CHAT_CARTRIDGE:-artifacts/steerer_chat_production_v3_strict_b384/chat_cartridge.pt}"
BASE_MODEL="${BASE_MODEL:-artifacts/steerer_v4/steerer_best_b.pt}"
GENERAL_STEERER="${GENERAL_STEERER:-artifacts/steerer_v4/steerer_best_b.pt}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_K="${TOP_K:-40}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
CONTEXT_LEN="${CONTEXT_LEN:-128}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.15}"
MAX_SENTENCES="${MAX_SENTENCES:-0}"

args=("$@")
if [[ ${#args[@]} -eq 0 ]]; then
  args=(--interactive)
fi

exec python hybrid/chat_cartridge.py \
  --base-model "$BASE_MODEL" \
  --general-steerer "$GENERAL_STEERER" \
  --chat-cartridge "$CHAT_CARTRIDGE" \
  --device "$DEVICE" \
  --mode chat \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --top-p "$TOP_P" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --context-len "$CONTEXT_LEN" \
  --repetition-penalty "$REPETITION_PENALTY" \
  --max-sentences "$MAX_SENTENCES" \
  "${args[@]}"