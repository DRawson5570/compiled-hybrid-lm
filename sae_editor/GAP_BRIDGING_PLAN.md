# NRTCS Gap-Bridging Plan

## Scope

Close all known gaps between the current NRTCS implementation and the full vision from `NRTCS_SPEC.md` and `USE_CASES.md`. This document is both a roadmap and a spec — it replaces open questions with concrete design decisions.

---

## Decisions

Made upfront to eliminate ambiguity.

| Decision | Rationale |
|----------|-----------|
| **Start with Architecture Abstraction (W1)** | Everything else — attention splicing, SAE integration, CLI — needs to know tensor names and layer layouts. W1 is the foundation. |
| **SAE Training (W3) in parallel with Attention (W2)** | They share no code. Both depend on W1. Both are blockers for the production demo. |
| **Attention scope: Qwen first, GPT-2 second** | Qwen2.5-1.5B is the primary model target. It uses separate Q/K/V/O projections (simpler than GPT-2's fused c_attn). The UCN backend already verified attention fidelity on Qwen at cosine=1.0. |
| **Production demo model: Qwen2.5-1.5B** | Already cached locally. Fast to load. Explicit Q/K/V/O projections. Verified attention fidelity. |
| **NRTCS DSL: thin layer over Python dict** | The NRTCS spec grammar (`layer N { override X = dense_map(...) }`) maps cleanly to our existing `{layer: {keys, values}}` dict format. No need to involve the UCN DSL (`mix`, `project`, `transform`). Those are separate concerns. |
| **No SAE training in CI** | SAE training takes hours per layer. CI tests use synthetic/random SAEs. Real SAE training is a one-time operation run manually or via a separate script. |

---

## Waves

### Wave 1: Architecture Abstraction

**Why first:** Every component that touches model weights (splicer, decompiler, future attention splicer, CLI) needs to know tensor names, shapes, and layer layouts. Currently all hardcoded to Qwen-style naming.

**New file:** `sae_editor/architectures.py`

```python
@dataclass
class ArchitectureSpec:
    """Complete tensor map for one model architecture."""
    name: str                           # "qwen2", "gpt2", "deepseek-custom"
    layer_prefix: str                   # "model.layers.{layer}"
    mlp_down_suffix: str                # "mlp.down_proj.weight"
    mlp_up_suffix: str                  # "mlp.up_proj.weight"
    mlp_gate_suffix: str | None         # "mlp.gate_proj.weight" or None (no gate)
    attn_q_suffix: str | None           # "self_attn.q_proj.weight" or None (fused QKV)
    attn_k_suffix: str | None
    attn_v_suffix: str | None
    attn_o_suffix: str
    mlp_type: Literal["simple", "gated"]     # simple = c_fc/c_proj, gated = SwiGLU
    attn_type: Literal["separate", "fused"]   # separate = Q/K/V/O, fused = c_attn + c_proj
    has_gqa: bool
    has_rope: bool
    layer_access_path: str              # "model.model.layers" | "transformer.h" | "layers"

    def mlp_down_name(self, layer: int) -> str: ...
    def mlp_up_name(self, layer: int) -> str: ...
    def attn_q_name(self, layer: int) -> str: ...
    def attn_k_name(self, layer: int) -> str: ...
    def attn_v_name(self, layer: int) -> str: ...
    def attn_o_name(self, layer: int) -> str: ...

    @classmethod
    def detect(cls, safetensors_path: str) -> "ArchitectureSpec": ...
    @classmethod
    def detect_from_keys(cls, keys: list[str]) -> "ArchitectureSpec": ...


# Built-in registry:
QWEN2 = ArchitectureSpec(...)
GPT2 = ArchitectureSpec(...)
DEEPSEEK_CUSTOM = ArchitectureSpec(...)
LLAMA3 = ArchitectureSpec(...)          # same pattern as Qwen2
MISTRAL = ArchitectureSpec(...)        # same pattern as Llama
```

**Modifications:**
- `splicer.py`: `splice_mlp` accepts `ArchitectureSpec` instead of `model_name: str`. The hardcoded `down_proj.weight` / `up_proj.weight` suffixes become lookups on the spec. Backward compatible: string input auto-detects or falls back to old behavior.
- `pipeline.py`: `splice_patches` and `round_trip` accept `ArchitectureSpec`. Default remains Qwen2.
- `decompiler.py`: `_get_layer` uses `arch.layer_access_path` for cleaner dispatch. Not required — the current fallback chain works — but cleaner.

**Tests (8 new):**
- `test_arch_name_construction`: Every built-in spec generates correct tensor names at layer 0 and layer N.
- `test_arch_detect_from_keys_gpt2`: Pass GPT-2 safetensors keys → returns GPT2 spec.
- `test_arch_detect_from_keys_qwen`: Pass Qwen safetensors keys → returns QWEN2 spec.
- `test_arch_mlp_type_gated_vs_simple`: Gate is only present for gated MLPs.
- `test_arch_attn_type_separate_vs_fused`: Separate Q/K/V only for separate attention.
- `test_splice_mlp_with_arch_spec`: Splice via ArchitectureSpec, verify correct tensor changed.
- `test_splice_mlp_with_gpt2_spec`: GPT-2 c_fc/c_proj naming works end-to-end.
- `test_splice_mlp_backward_compat`: String `model_name` parameter still works.

**Verification:** All 73 existing tests pass. New tests pass. Manual: create a GPT-2 safetensors file, splice via `splice_mlp(layer=0, W_down, W_up, arch=GPT2)`, verify correct tensor was modified.

---

### Wave 2: Attention Support

**Depends on:** Wave 1 (Architecture).

**Current state:** The UCN `ReferenceBackend._apply_full_attention` compiles full multi-head attention (Q/K/V/O + RoPE + GQA + causal softmax + output projection) at **cosine=1.000000** fidelity against Qwen2.5-1.5B layer 0. The decompiler already captures layer activations. The recompiler can compile key-value pairs into W_down/W_up. The splicer can write tensors to safetensors.

**What's missing:** The pipeline that wires all four together for attention weights specifically.

**New file:** `sae_editor/attention.py`

```python
from sae_editor.architectures import ArchitectureSpec

class AttentionExtractor:
    """Extract attention weights from a loaded model layer into a PrimitiveEntry."""
    
    def __init__(self, arch: ArchitectureSpec):
        self.arch = arch
    
    def extract(self, model, layer: int) -> tuple[dict[str, Tensor], dict]:
        """Returns ({W_q, W_k, W_v, W_o, ...bias...}, metadata_dict).
        
        For fused QKV architectures (GPT-2): splits c_attn into Q/K/V slices.
        For separate architectures (Qwen): reads q_proj, k_proj, v_proj, o_proj directly.
        Metadata includes: n_heads, n_kv_heads, head_dim, has_rope, cos_sin_cache.
        """
    
    def to_stdlib_entry(self, weights: dict, metadata: dict, layer: int) -> PrimitiveEntry:
        """Convert extracted weights + metadata to a UCN PrimitiveEntry.
        
        Sets operator_type="multihead_attention", populates weight_data.
        """


class AttentionSplicer:
    """Splice attention weights into a safetensors file."""
    
    def __init__(self, arch: ArchitectureSpec):
        self.arch = arch
    
    def splice(self, safetensors_path: str, layer: int,
               W_q, W_k, W_v, W_o,
               b_q=None, b_k=None, b_v=None, b_o=None):
        """Write Q/K/V/O weights (and optional biases) to safetensors.
        
        For fused architectures (GPT-2): concatenates Q/K/V into c_attn before writing.
        Shape validation per architecture spec.
        """
    
    def transplant(self, safetensors_path: str, 
                   source_layer: int, target_layer: int):
        """Copy attention from one layer to another (cross-layer transplant).
        Reads source, splices into target. Verifies shapes match.
        """
```

**Modifications:**
- `splicer.py`: `SafetensorsSplicer` adds `splice_attention(layer, AttentionSplicer)`. The internal `splice_mlp` stays — both coexist.
- `pipeline.py`: `NRTCSPipeline` adds `compile_attention_from_extraction(model, layer)` → returns weight dict ready for splicing. Wraps `AttentionExtractor.extract()`.
- `cli.py`: New subcommand: `nrtcs extract-attention model.safetensors --layer 0 --output layer0_attn/`

**Tests (12 new):**
- `test_extract_qwen_attention_shapes`: Extract Q/K/V/O from Qwen2.5-1.5B layer 0, verify shapes match config (n_heads=12, n_kv_heads=2, head_dim=128).
- `test_extract_gpt2_attention_split`: Extract Q/K/V from GPT-2 (fused c_attn), verify split produces correct slices.
- `test_extract_metadata`: Verify n_heads, n_kv_heads, head_dim are correctly extracted.
- `test_to_stdlib_entry_round_trip`: extract → to_stdlib_entry → load weights → verify matches original.
- `test_splice_attention_qwen`: Save Qwen to temp safetensors → extract attention → splice back → verify tensors match.
- `test_splice_attention_gpt2`: Same for GPT-2 fused c_attn/proj.
- `test_splice_attention_changes_identity`: Splice identical weights → reload → verify output unchanged.
- `test_splice_attention_changes_random`: Splice random weights → reload → verify output changed.
- `test_splice_attention_transplant`: Extract layer 0 attention → splice into layer 2 → verify layer 2 changed, layer 0 untouched.
- `test_attention_extractor_invalid_arch`: Raises error for unsupported attention type.
- `test_attention_fused_split_reconstruction`: Split GPT-2 c_attn into Q/K/V → reconstruct → matches original c_attn byte-for-byte.
- `test_end_to_end_attention_compile_splice`: Full pipeline: extract → compile to PrimitiveEntry → splice → verify.

**Verification:** The UCN `_apply_full_attention` already proved cosine=1.0 fidelity. These tests verify the NRTCS-side I/O (extraction, splicing, cross-layer transplant) — not that the attention computation itself is correct (that's UCN's responsibility).

**Scope note:** This wave targets Qwen2.5-1.5B and GPT-2 (small). Adding Llama, Mistral, DeepSeek custom architectures means adding entries to `architectures.py` — no new code needed in `attention.py`.

---

### Wave 3: SAE Training Pipeline

**Depends on:** Wave 1 (Architecture, for layer access).  
**Runs in parallel with:** Wave 2.

**Current state:** `full_compiled_experiment/ucn/decompile/sae.py` has `train_sae()` and `train_sae_on_layer_activations()`. `find_and_extract_copy_head.py` demonstrates the end-to-end flow. But there's no library-level API for "train SAEs on layers [0,2,5,8,14] and register them with the decompiler."

**New file:** `sae_editor/sae_training.py`

```python
class SAETrainingPipeline:
    """Train SAEs on model layers, ready for NRTCSDecompiler."""
    
    def __init__(self, arch: ArchitectureSpec):
        self.arch = arch
    
    def collect_layer_activations(self, model, tokenizer, 
                                   texts: list[str], layers: list[int],
                                   max_length=128, batch_size=4
                                   ) -> dict[int, Tensor]:
        """Collect residual stream activations for specified layers."""
    
    def train_all(self, model, tokenizer, texts: list[str],
                  layers: list[int], n_features=256, steps=2000, lr=1e-3
                  ) -> dict[int, SparseAutoencoder]:
        """Train one SAE per layer. Returns {layer_idx: trained_sae}."""


class SAERegistry:
    """Persist and load trained SAEs with metadata."""
    
    DIR_LAYOUT = "saes/{model_name}/layer_{idx}.pt"
    
    def save(self, saes: dict[int, SparseAutoencoder], path_prefix: str): ...
    def load(self, path_prefix: str) -> dict[int, SparseAutoencoder]: ...
    def create_decompiler(self, model, tokenizer, threshold=0.1) -> NRTCSDecompiler:
        """Convenience: load() + NRTCSDecompiler() in one call."""

# Training script (separate from test suite):
# sae_editor/scripts/train_saes.py
#   python -m sae_editor.scripts.train_saes \
#       --model Qwen/Qwen2.5-1.5B \
#       --layers 0,2,5,8,14 \
#       --texts training_corpus.txt \
#       --output saes/qwen2.5-1.5b/
```

**Tests (6 new):**
- `test_train_on_synthetic_model`: Synthetic model (d_model=64), 2 layers, 50 sample texts → verify SAE training completes.
- `test_save_load_round_trip`: Train → save → load → verify encoder/decoder weights match.
- `test_registry_create_decompiler`: SAERegistry.load() → create_decompiler() → decompiler collects activations correctly.
- `test_pipeline_collects_correct_shapes`: Collected activations have shape (N_texts, max_length, d_model).
- `test_train_all_multiple_layers`: 3 layers trained in sequence, no cross-contamination.
- `test_smoke_tiny_gpt2`: Train one SAE on tiny-gpt2 (20 texts, 100 steps) → extract features → verify feature_indices is non-empty.

**Verification:** The SAE training code from `full_compiled_experiment` is battle-tested. These tests verify the wrapping, persistence, and decompiler integration — not the SAE quality itself.

---

### Wave 4: Decompiler→Recompiler Automation

**Depends on:** Wave 3 (trained SAEs), Wave 1 (Architecture).

**Current state:** `extract_features()` returns feature indices, vectors, activation strengths. User manually selects features and creates key-value dictionary. No automated mapping from "the concept of France" to a specific SAE feature.

**New file:** `sae_editor/circuit_editor.py`

```python
class CircuitEditor:
    """Bridge from decompiled features to recompilable edits."""
    
    def __init__(self, decompiler: NRTCSDecompiler):
        self.decompiler = decompiler
    
    def find_feature_activating_on(self, texts: list[str], top_k=5
                                   ) -> dict[int, list[int]]:
        """Find which SAE features activate most strongly on given texts.
        
        Returns {layer_idx: [feature_indices]} sorted by activation strength.
        """
    
    def extract_feature_vector(self, layer: int, feature_idx: int
                               ) -> Tensor:
        """Return the decoder vector for a specific feature (direction in d_model)."""
    
    def extract_value_vector_for_text(self, text: str, layer: int
                                      ) -> Tensor:
        """Run the model on 'text', capture activation at 'layer', 
        return the residual stream vector at the last token position.
        This is the 'value' side of a key-value edit.
        """
    
    def create_edit_from_texts(self, source_text: str, target_text: str,
                                layer: int, top_k=1
                                ) -> dict[int, dict[str, Tensor]]:
        """High-level: 'Fix association from source to target.'
        
        1. Find features active on source_text
        2. For top_k features: extract feature vector (key) + target's last-token activation (value)
        3. Return {layer: {"keys": (k, d), "values": (k, d)}}
        
        This is the France→Paris in one call.
        """
    
    def verify_edit(self, edits: dict, model, tokenizer, 
                    test_prompt: str) -> bool:
        """After splicing, verify the edit had the intended effect.
        
        Runs model on test_prompt, checks output token distribution.
        Returns True if behavior changed in the expected direction.
        """
```

**Tests (8 new):**
- `test_find_feature_on_synthetic`: Synthetic model + random SAE → inject known feature direction → verify `find_feature_activating_on` returns that feature.
- `test_extract_feature_vector_shape`: Vector has shape (d_model,).
- `test_extract_value_vector_shape`: Last-token activation has shape (d_model,).
- `test_create_edit_structure`: Returns dict with "keys" and "values", correct shapes.
- `test_create_edit_orthogonal_keys`: Keys are reasonably distinct (dot product < 0.9 between consecutive keys).
- `test_verify_edit_detects_change`: After identity splice, verify_edit returns True (no degradation). After random splice, verify_edit returns False (behavior changed).
- `test_smoke_france_paris_on_tiny_gpt2`: Run create_edit_from_texts("London", "Paris") on tiny-gpt2 → compile → verify edit is structurally valid (reconstruction error < threshold).
- `test_find_feature_no_activation_returns_empty`: Empty/irrelevant text returns no features.

---

### Wave 5: NRTCS DSL Integration

**Depends on:** Nothing. Independent. Can run in parallel with any other wave.

**Current state:** Phase 2 refactoring is done via Python API. The NRTCS spec (§3) defines a text DSL:
```
layer 14 {
    override mlp.down_proj = dense_map(
        < [0.9, -0.1, 0.1, 0.1], [0.1, 0.1, 0.9, 0.1] >
    );
}
```

**Design decision:** The NRTCS DSL compiles directly to our Python dict format `{layer: {keys, values}}`. It does NOT compile to the UCN DSL (`mix`, `project`, `transform`). Those are separate concerns — UCN compiles programs from scratch, NRTCS DSL describes patches to existing models.

**New file:** `sae_editor/dsl/__init__.py` + `sae_editor/dsl/nrtcs_parser.py`

```python
def parse_nrtcs(source: str) -> dict[int, dict[str, Tensor]]:
    """Parse NRTCS DSL text into edit dict.
    
    Grammar (from NRTCS_SPEC.md §3.1):
        layer N { override ID = dense_map(< key_vec, value_vec >, ...); }
    
    Returns {N: {"keys": (N_pairs, d_in), "values": (N_pairs, d_out)}}
    """

def serialize_nrtcs(edits: dict[int, dict[str, Tensor]]) -> str:
    """Serialize edit dict back to NRTCS DSL text.
    
    Useful for version control, code review, and agent-to-agent communication.
    """
```

**Modifications:**
- `cli.py`: New subcommand: `nrtcs parse file.uvm --output edits.pt` and `nrtcs serialize edits.pt --output file.uvm`
- `pipeline.py`: `NRTCSPipeline.compile_from_uvm_file(path)` — parses DSL file, compiles, returns patches.

**Tests (10 new):**
- `test_parse_single_layer_dense_map`: Parse a valid DSL string → correct dict structure.
- `test_parse_multiple_layers`: Two `layer N { ... }` blocks → two dict entries.
- `test_parse_multiple_pairs`: `dense_map(< k1, v1 >, < k2, v2 >)` → 2 key-value pairs.
- `test_parse_float_literals`: Vector literals with negative numbers, decimals.
- `test_serialize_round_trip`: parse(serialize(edits)) == edits (numerically, atol=1e-6).
- `test_parse_empty_program`: No layers → returns {}.
- `test_parse_rejects_invalid_syntax`: Missing semicolon, unknown keyword → raises ParseError.
- `test_cli_parse_round_trip`: CLI parse → save → CLI serialize → reload → identical.
- `test_compile_from_uvm_file`: Full pipeline from DSL file to compiled patches.
- `test_smoke_france_paris_dsl`: Parse the France→Paris example from NRTCS_SPEC.md §6 → compile → verify reconstruction.

---

### Wave 6: Production Integration Demo

**Depends on:** Waves 1, 2, 3, 4.

**What it is:** A single script that demonstrates the complete vision — NRTCS patches applied to a model, then the compiled features + steerer cartridge run on top of the patched model.

**New file:** `sae_editor/demo_full_pipeline.py`

```python
"""
Full NRTCS pipeline demo.

Usage:
    python -m sae_editor.demo_full_pipeline \
        --model Qwen/Qwen2.5-1.5B \
        --patch "The capital of France is Paris." \
        --output patched_model.safetensors

Flow:
    1. Load model, train SAEs on layers [0, 2, 5, 8, 14] (or load cached)
    2. Decompile: find "France" feature in relevant layer
    3. Extract: get France key vector, Paris value vector
    4. Create edit dict, compile (with crosstalk prevention)
    5. Splice into patched_model.safetensors
    6. Load patched model, verify output shifted
    7. Load compiled features + steerer cartridge on top
    8. Run benchmark, compare patched+cartridge vs unpatched+cartridge
"""
```

**Not a test** — it's a demo script that runs end-to-end. It's gated behind `--run` flag (default is dry-run showing what WOULD happen).

**Verification:** Manual run on Qwen2.5-1.5B. Records results in `EXPERIMENT_LOG.md`.

---

### Wave 7: Remaining Gaps

Small, independent tasks. No blocking dependencies.

#### G5: Cross-Architecture Transfer

**New file:** `sae_editor/transfer.py`

```python
def project_features(features: Tensor, from_d_model: int, to_d_model: int
                    ) -> Tensor:
    """Project feature vectors from one d_model space to another."""
    # Uses random projection (Johnson-Lindenstrauss) or learned linear map
    # Returns features resized to to_d_model

def transfer_edit(edits: dict, source_arch: ArchitectureSpec,
                  target_arch: ArchitectureSpec) -> dict:
    """Convert an edit dict from one architecture to another."""
```

**Tests (3):** Projection preserves pairwise distances. Transfer produces structurally valid edit dict. Smoke test: GPT-2 → tiny synthetic model.

#### G7: Dimension Compaction

**Add to:** `sae_editor/recompiler.py`

```python
def compact_features(W_down: Tensor, n_components: int
                    ) -> tuple[Tensor, Tensor]:
    """PCA compaction of W_down columns when null space is low.
    
    Returns (W_compacted, basis) where basis is the PCA transform.
    Use decompact_features() to reconstruct.
    """
```

**Add to:** `RecompilerEngine.compile()` — auto-triggers if `compute_null_space_rank() < 0.10 * d_model`.

**Tests (3):** Compaction reduces rank. Decompaction recovers original within tolerance. Auto-trigger at 10% threshold.

#### G8: Non-Linear Activation Mitigation

**Add to:** `sae_editor/recompiler.py`

```python
def pre_activation_scale(keys: Tensor, activation: str = "gelu",
                         target_range: float = 2.0) -> Tensor:
    """Scale key vectors so their activations stay in the linear region.
    
    For GeLU: linear region is approximately [-2, 2].
    Scales keys such that max(abs(key_i @ x)) < target_range for expected x.
    """
```

**Tests (2):** Scaled keys are within target range. Reconstruction fidelity is preserved after scaling.

#### G9: CLI Completion

**Modify:** `cli.py` — `cmd_decompile`

```python
def cmd_decompile(args):
    arch = ArchitectureSpec.detect_from_model_name(args.model_path)
    saes = SAERegistry.load(args.sae_path)
    decompiler = NRTCSDecompiler(model, tokenizer, saes)
    features = decompiler.extract_features(args.texts)
    torch.save(features, args.output)
```

**Tests (1):** `cmd_decompile` with synthetic model + synthetic SAEs produces output file.

---

## Summary

| Wave | New files | Modified files | New tests | Cumulative tests | Estimated effort |
|------|-----------|----------------|-----------|------------------|-----------------|
| W1: Architecture | 1 | 3 | 8 | 81 | 3-4 hrs |
| W2: Attention | 1 | 2 | 12 | 93 | 6-8 hrs |
| W3: SAE Training | 1 | 0 | 6 | 99 | 3-4 hrs |
| W4: Automation | 1 | 0 | 8 | 107 | 3-4 hrs |
| W5: DSL | 2 | 1 | 10 | 117 | 4-5 hrs |
| W6: Demo | 1 | 0 | 0 | 117 | 2-3 hrs |
| W7: Remaining | 4 small | 2 | 11 | 128 | 2-3 hrs |
| **Total** | **11** | **8** | **55** | **128** | **23-31 hrs** |

### Execution order

```
W1 ────────► W2 ──┐
                    ├──► W6
W1 ────────► W3 ──► W4 ──┘

W5 ──────────── (parallel, independent)

W7 ──────────── (parallel, after W1, independent)
```

### After completion

The system covers every use case from `USE_CASES.md`:
- **Model surgery:** CircuitEditor automates find → extract → edit → compile → splice.
- **Safety gates:** NRTCS DSL `splice SAFETY_GATE = gate(...)` parsed, compiled, spliced.
- **Model variants:** ArchitectureSpec enables one base model → N patched variants across architectures.
- **Attention transplantation:** AttentionExtractor + AttentionSplicer = full attention head decompile→recompile.
- **Research loop:** Decompile → hypothesize → CircuitEditor.edit → splice → verify. No gradient training.
- **Cross-architecture:** transfer.py projects features between d_model spaces.
- **Production:** demo_full_pipeline.py shows NRTCS + compiled features + steerer working together.
