# Neurosymbolic Round-Trip Compilation Stack (NRTCS)
## System Specification V4.0

**Document Reference:** NRTCS-CORE-SPEC-V4.0  
**Classification:** Systems Engineering & Compiler Architecture Specification  
**Status:** Approved Reference Standard  

---

## 1. System Architecture & Life Cycle

The **Neurosymbolic Round-Trip Compilation Stack (NRTCS)** defines a closed-loop system for the decompilation, symbolic refactoring, and recompilation of neural network parameters. The system enables deterministic modification of neural behaviors without gradient-based training.

```
       ┌────────────────────────────────────────────────────────┐
       │             1. DECOMPILER ENGINE (C2S)                 │
       │  - Reads: source.safetensors                           │
       │  - Extracts: Circuits, Features, Factual Associations  │
       │  - Outputs: model_source.uvm (UVM-DSL AST)             │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             2. REFACTORING & SPLICING INTERFACE        │
       │  - Developer / Agent modifies model_source.uvm         │
       │  - Injects deterministic safety-gates, corrects facts  │
       │  - Outputs: patched_source.uvm                         │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             3. RECOMPILER ENGINE (S2C)                 │
       │  - Compiles: UVM-DSL instructions into active matrices │
       │  - Applies: Crosstalk Prevention & Subspace Isolation  │
       │  - Outputs: compiled_patch.bin                         │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             4. BINARY SPLICING ENGINE                  │
       │  - Merges: compiled_patch.bin into source.safetensors  │
       │  - Outputs: patched_model.safetensors                  │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Phase I: Decompiler Engine (Continuous to Symbolic - C2S)

The C2S Decompiler converts continuous matrix parameters in a `.safetensors` file into symbolic instructions in the UVM-DSL format.

### 2.1 Feature Disentanglement Pipeline
The decompiler runs a collection of pre-trained Sparse Autoencoders (SAEs) over the target layer activations. It maps active neural pathways to discrete feature indices.

```
               [ Layer Activation Stream x ]
                             │
                             ▼
               ┌───────────────────────────┐
               │  Sparse Autoencoder (SAE) │
               └─────────────┬─────────────┘
                             │
               (Features with h(x) > Threshold)
                             │
                             ▼
               ┌───────────────────────────┐
               │    Sparsified Features    │
               │   (Symbolic Dictionary)   │
               └───────────────────────────┘
```

The activation of feature $i$ is extracted when:

$$h_i(\mathbf{x}) = \max\left(0, \mathbf{w}_i^T \mathbf{x} + b_i\right) > \tau$$

Where:
*   $\mathbf{w}_i$ is the $i$-th encoder direction.
*   $\tau$ is the activation threshold (e.g., $\tau = 0.1$).

### 2.2 Circuit Extraction (Attribution Patching)
To map how extracted features flow together into functional circuits, the decompiler executes **Path Attribution Patching**. For a target downstream feature activation $y$, the attribution $A$ of an upstream circuit pathway from feature $x_j$ is computed as:

$$A(x_j \rightarrow y) = \left( \mathbf{x}_j \cdot \nabla_{\mathbf{x}_j} y \right)$$

If $A(x_j \rightarrow y) > \theta_{\text{attribution}}$, the decompiler registers a dependency link and outputs a corresponding UVM-DSL step.

---

## 3. Phase II: Symbolic Refactoring Language (UVM-DSL v4.0)

UVM-DSL v4.0 is a structured programming language representing the decompiled network. It allows developers to specify precise layer overrides, hot-patches, and memory-mapping updates.

### 3.1 Grammar Specification
```bnf
<patch_program>   ::= <layer_declaration_list> <execution_block>
<layer_declaration_list> ::= <layer_decl> | <layer_decl> <layer_declaration_list>
<layer_decl>      ::= "layer" <integer> "{" <primitive_list> "}"

<primitive_list>  ::= <primitive> | <primitive> ";" <primitive_list>
<primitive>       ::= "override" <primitive_id> "=" <expr>
                    | "splice" <primitive_id> "at" <coordinate_range>

<expr>            ::= "dense_map" "(" <key_value_pairs> ")"
                    | "orthogonal_projection" "(" <subspace_ref> ")"
                    | "gate" "(" <id> "," <threshold> ")"

<coordinate_range> ::= "[" <integer> ":" <integer> "]"
<key_value_pairs>  ::= "<" <vector_literal> "," <vector_literal> ">" 
                     | "<" <vector_literal> "," <vector_literal> ">" "," <key_value_pairs>
```

### 3.2 Code Example: Correcting a Factual Association
The following UVM-DSL source patch overrides a faulty country-capital lookup in layer 14:

```python
# patched_source.uvm
layer 14 {
    # 1. Target the specific memory projection partition
    override PRM_0x8F01 = dense_map(
        # Replace faulty Paris lookup with verified coordinates
        < [0.12, -0.44, 0.89, -0.01, 0.33, 0.55, -0.12, 0.22], [0.88, -0.11, 0.34, 0.90, -0.45, 0.12, 0.76, -0.01] >
    );
    
    # 2. Inject a safety gate to intercept toxic content in a specific subspace
    splice PRM_SAFETY_GATE = gate(workspace_reg_3, threshold=0.85) at [64:128];
}
```

---

## 4. Phase III: Recompiler Engine (Symbolic to Continuous - S2C)

The S2C Recompiler translates the patched UVM-DSL program back into numerical matrices ($W_Q, W_K, W_V, W_{\text{FFN}}$).

```
                      [ Patched UVM-DSL Program ]
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │ 1. Matrix Construction  │
                      └────────────┬────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │ 2. Crosstalk Prevention │
                      └────────────┬────────────┘
                                   │
                                   ▼
                      [ Compiled Matrix Patch Binary ]
```

### 4.1 Analytical Matrix Construction
When recompiling a `dense_map` memory structure, the compiler takes $N$ key vectors $\mathbf{K} = [\mathbf{k}_1, \dots, \mathbf{k}_N]^T \in \mathbb{R}^{N \times D_{\text{in}}}$ and $N$ value vectors $\mathbf{V} = [\mathbf{v}_1, \dots, \mathbf{v}_N]^T \in \mathbb{R}^{N \times D_{\text{out}}}$. 

To compile these pairs into FFN matrices $W_{\text{down}}$ and $W_{\text{up}}$, the compiler resolves them analytically:

$$W_{\text{down}} = \mathbf{K}^T \left( \mathbf{K} \mathbf{K}^T \right)^{-1}$$

$$W_{\text{up}} = \mathbf{V}^T$$

This formulation guarantees that when input key $\mathbf{k}_i$ is processed, it maps to value $\mathbf{v}_i$:

$$\mathbf{k}_i W_{\text{down}} W_{\text{up}} = \mathbf{v}_i$$

### 4.2 Crosstalk Prevention Pass (Orthogonal Subspace Projection)
To ensure the compiled weights do not interfere with unmodified parts of the model (crosstalk), the compiler executes an **Orthogonal Projection Pass**.

The compiler identifies the null space of the original model's active features $\mathbf{U}$ and projects the new, compiled parameters $W_{\text{compiled}}$ into this orthogonal subspace:

$$\mathbf{P}_{\perp} = \mathbf{I} - \mathbf{U}(\mathbf{U}^T\mathbf{U})^{-1}\mathbf{U}^T$$

$$W_{\text{final}} = \mathbf{P}_{\perp} W_{\text{compiled}}$$

This ensures that the compiled parameters only trigger in the targeted activation zones, preventing unexpected degradation in unrelated benchmarks.

---

## 5. Phase IV: Binary Splicing & Serialization Engine

The Splicing Engine executes zero-copy insertion of the compiled matrices directly into the source `.safetensors` file.

### 5.1 Safetensors Block Replacement Protocol
Because `.safetensors` uses a flat, uncompressed memory layout with an initial JSON header, the Splicing Engine can replace tensor payloads inline without re-writing the entire file, provided the shapes match.

```
+------------------------------------------------------------+
|                  Original Safetensors File                 |
|                                                            |
|  [JSON Metadata Header]  [Tensor 1 Payload]  [Tensor 2]... |
+------------------------------------------------------------+
                                  ▲
                                  │ (Inline Pointer Swap)
                                  ▼
+------------------------------------------------------------+
|                    Compiled Patch Binary                   |
|                                                            |
|                  [Replacement Tensor Payload]              |
+------------------------------------------------------------+
```

```python
# Conceptual Inline Splicer implementation
import os
import json
import mmap

def splice_tensor_inline(safetensors_path, tensor_name, new_payload_bytes):
    with open(safetensors_path, "r+b") as f:
        # 1. Map file to memory
        mm = mmap.mmap(f.fileno(), 0)
        
        # 2. Parse JSON header length
        header_len = int.from_bytes(mm[0:8], byteorder="little")
        header_bytes = mm[8:8+header_len]
        header = json.loads(header_bytes.decode("utf-8"))
        
        # 3. Locate offsets for the targeted tensor
        tensor_meta = header[tensor_name]
        start_offset = 8 + header_len + tensor_meta["data_offsets"][0]
        end_offset = 8 + header_len + tensor_meta["data_offsets"][1]
        
        # 4. Assert size equivalence to maintain file alignment
        expected_size = end_offset - start_offset
        assert len(new_payload_bytes) == expected_size, "Spliced tensor size must match original."
        
        # 5. Inline memory swap
        mm[start_offset:end_offset] = new_payload_bytes
        mm.flush()
        print(f"Tensor '{tensor_name}' successfully hot-patched inline.")
```

---

## 6. Concrete Verification Walkthrough

This scenario outlines how to patch a factual error in Layer 2 of a toy model using the NRTCS pipeline.

### 1. Source State:
*   **Prompt:** `"The capital of France is [mask]"`
*   **Halucinated Output:** `"London"`

### 2. Execution of Decompiler (`c2s`):
The decompiler runs over `model.safetensors` and identifies the factual routing matrix `model.layers.2.mlp.down_proj`.
It extracts the feature coordinates:
*   Key vector for `"France"`: `[0.9, -0.1, 0.1, 0.1]`
*   Faulty output vector: `[0.1, 0.9, 0.1, 0.1]` (which maps to `"London"`)

### 3. Symbolic Refactoring (`patched.uvm`):
The developer modifies the UVM source file to point to the correct output vector representation for `"Paris"`:

```python
# patched.uvm
layer 2 {
    override mlp.down_proj = dense_map(
        < [0.9, -0.1, 0.1, 0.1], [0.1, 0.1, 0.9, 0.1] > # Correct target vector
    );
}
```

### 4. Recompiler execution (`s2c`):
The compiler processes the UVM file, calculates the new projection matrices, runs the orthogonal projection pass to protect surrounding factual associations, and outputs the patched binary payload.

### 5. Splicing Execution:
The compiler runs `splice_tensor_inline("model.safetensors", "model.layers.2.mlp.down_proj", patch_bytes)`. The binary payload is swapped inline in memory.

### 6. Verification:
*   **Prompt:** `"The capital of France is [mask]"`
*   **Patched Output:** `"Paris"`
*   **Collateral Verification:** Other capital city queries remain unaffected due to the orthogonal projection pass.

---

## 7. System Constraints & Risk Mitigations

### 7.1 Dimension Saturation
*   **Risk:** As more symbolic patches are compiled, the available orthogonal null space in the model's residual stream diminishes, leading to performance degradation on general benchmarks.
*   **Mitigation:** The recompiler tracks the remaining dimensions of the null space $\mathbf{P}_{\perp}$. If the available rank falls below 10% of $D_{\text{model}}$, the compiler alerts the operator that a **Dimension Compaction Pass** (using principal component analysis) is required to free up workspace variables.

### 7.2 Non-Linear Activation Collisions
*   **Risk:** The analytical construction assumes linear operations. Non-linear activation functions (e.g., GeLU) can introduce unexpected scaling or clamping of compiled vectors.
*   **Mitigation:** The compiler applies pre-activation scaling. It scales the generated key matrices such that key activations sit in the strictly linear regions of the target activation function during execution.
