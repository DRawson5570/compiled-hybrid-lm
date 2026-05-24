"""config.py — CMI repository root configuration.

Resolves all paths relative to the repo root, not hardcoded user directories.
Import this first in any script to set up sys.path correctly.
"""
from pathlib import Path
import sys

# Root of the hybrid_steering repository
REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure REPO_ROOT is at the front of sys.path for all imports
sys.path.insert(0, str(REPO_ROOT))

# Artifact directories (relative to REPO_ROOT)
ARTIFACTS = REPO_ROOT / "artifacts"
DATA_WIKITEXT = ARTIFACTS / "wikitext_gpt2"
DATA_CODE = ARTIFACTS / "code_steerer"
COMPILED_PRIORS = ARTIFACTS / "compiled_priors_v3"
CHECKPOINTS_C4 = ARTIFACTS / "c4_v2_768_x30"

# Default model checkpoint
DEFAULT_NEURAL_CKPT = CHECKPOINTS_C4 / "best.pt"
DEFAULT_TRAIN_IDS = DATA_WIKITEXT / "train_ids.pt"
DEFAULT_VAL_IDS = DATA_WIKITEXT / "validation_ids.pt"

# Output directories for trained artifacts
STEERER_V1_OUT = ARTIFACTS / "steerer_stream"
STEERER_V2_OUT = ARTIFACTS / "steerer_v2"
STEERER_V4_OUT = ARTIFACTS / "steerer_v4"
STEERER_CODE_OUT = ARTIFACTS / "steerer_code"
MODEL_340M_OUT = ARTIFACTS / "c4_340m"
