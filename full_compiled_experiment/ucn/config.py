import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS = os.path.join(REPO_ROOT, "..", "..", "artifacts", "ucn")
STDLIB_DIR = os.path.join(REPO_ROOT, "stdlib_weights")
CACHE_DIR = os.path.join(ARTIFACTS, "jit_cache")

DEFAULT_DTYPE = "float32"
DEFAULT_DEVICE = "cuda"
