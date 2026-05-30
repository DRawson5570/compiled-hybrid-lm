# System Specification: Neural-Symbolic Compiled Attention (NSCA)

**Document Reference:** NSCA-REF-SPEC-V1.0  
**Status:** Proposed Reference Architecture  

This document provides a technical specification for **Neural-Symbolic Compiled Attention (NSCA)**, an alternative architecture to standard static-weight attention layers. NSCA replaces traditional, dense, content-addressable attention projections with a dynamic, context-aware compilation pipeline. 

The architecture consists of a learned **Frontend Program Synthesizer** that maps input context sequences to programs in a vector-manipulation domain-specific language (DSL), and a **Backend JIT Compiler** that optimizes and compiles these programs into hardware-efficient kernels on the fly.

---

## 1. System Architecture Overview

Standard multi-head attention (MHA) performs static projection operations ($\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, $\mathbf{V} = \mathbf{X}\mathbf{W}_V$) followed by a dense dot-product softmax routing. 

NSCA processes inputs through three main phases:

```
[ Input Tokens: X ]
       │
       ▼
┌────────────────────────────────────────┐
│ 1. FRONTEND COMPILER (Synthesizer)     │
│  - Context Analyzer (Lightweight Net)  │ ---> Dynamic Generation
│  - Circuit Template Selector           │      of Vector DSL Program
└────────────────────────────────────────┘
       │
       ▼
  [ Vector DSL Program (AST) ]
       │
       ▼
┌────────────────────────────────────────┐
│ 2. BACKEND COMPILER (JIT Engine)       │
│  - Program Hash Cache Look-up          │ ---> Fused Triton/CUDA Kernels
│  - MLIR-Style Lowering & Fusion        │      or Optimized CPU Vectors
└────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ 3. EXECUTION ENGINE                    │
│  - Hardware-Accelerated Run            │ ---> Transformed Embeddings: Y
└────────────────────────────────────────┘
```

1. **The Frontend Compiler (Program Synthesizer):** Analyzes the context window $\mathbf{X} \in \mathbb{R}^{T \times d}$ and synthesizes a structured program in a Vector Manipulation DSL (**VM-DSL**).
2. **The Backend Compiler (JIT Engine):** Lowers the AST of the synthesized program into an intermediate representation (IR), applies optimization passes (fusion, dead-code elimination, sparse optimization), checks a compilation cache, and compiles a hardware-optimized kernel.
3. **The Execution Engine:** Executes the compiled kernel on the high-dimensional input vectors, producing the updated sequence representation $\mathbf{Y} \in \mathbb{R}^{T \times d}$.

---

## 2. Vector Manipulation DSL (VM-DSL) Specification

The VM-DSL is a strongly typed, deterministic language designed to express linear and non-linear transformations on high-dimensional vector spaces. It restricts the programming space to ensure that all programs can be compiled to hardware-efficient loops.

### 2.1 Type System
* `Vector<D>`: A high-dimensional vector of size $D$ (typically $D = d_{\text{model}}$).
* `Subspace<K>`: A reference to a slice of dimensions of size $K$ ($K \le D$).
* `Scalar`: A single-precision floating-point value.
* `Index`: An integer specifying a token index or position.

### 2.2 Grammar (BNF)
```bnf
<program>    ::= <statement_list>
<statement_list> ::= <statement> | <statement> ";" <statement_list>
<statement>  ::= <id> "=" <expr>
<expr>       ::= "project" "(" <id> "," <subspace> ")"
               | "rotate" "(" <id> "," <angle_expr> "," <axis_id> ")"
               | "mix" "(" <vector_list> "," <weight_list> ")"
               | "scale_shift" "(" <id> "," <scalar_expr> "," <scalar_expr> ")"
               | "sparse_route" "(" <id> "," <expert_list> "," <weight_list> ")"
               | "gather_context" "(" <id> "," <index_list> ")"

<vector_list> ::= "[" <id_list> "]"
<weight_list> ::= "[" <scalar_list> "]"
<id_list>     ::= <id> | <id> "," <id_list>
<scalar_list> ::= <scalar_expr> | <scalar_expr> "," <scalar_list>
```

### 2.3 Semantic Primitives
* **`project(v, S)`**: Projects vector `v` onto the coordinates defined by subspace `S`. Unused dimensions are treated as zero.
* **`rotate(v, theta, axis)`**: Applies a rotation of angle `theta` in the plane defined by `axis`. This implements selective rotary position embeddings (RoPE) or semantic rotations.
* **`mix(V, W)`**: Computes a weighted sum $\sum_i w_i v_i$.
* **`scale_shift(v, alpha, beta)`**: Applies element-wise affine transformation $\alpha \mathbf{v} + \beta$.
* **`sparse_route(v, experts, W)`**: Evaluates conditional execution pathways, routing `v` to a subset of predefined operations.

---

## 3. Frontend Compiler Specification (Synthesizer)

The Frontend Compiler maps the continuous input context $\mathbf{X}$ to a symbolic program. To avoid the high latency of freeform code generation, the Frontend uses a **Template-Based Synthesis** approach.

```
       [ Context Sequence: X ]
                  │
                  ▼
       ┌─────────────────────┐
       │  Context Embedder   │ ───> Latent Space Representation (z)
       └─────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│   Template    │   │   Parameter   │
│   Selector    │   │   Generator   │
└───────────────┘   └───────────────┘
        │                   │
  (Selects AST)       (Fills scalar
        │              coefficients)
        └─────────┬─────────┘
                  ▼
         [ Executable AST ]
```

### 3.1 Context Analyzer
The Context Analyzer is a small, parametric neural module with parameters $\theta_c$. Given a sequence $\mathbf{X} \in \mathbb{R}^{T \times d}$, it extracts a latent state $\mathbf{z}$:

$$\mathbf{z} = \text{Pooling}(\text{MLP}(\text{Attention}_{\text{light}}(\mathbf{X}))), \quad \mathbf{z} \in \mathbb{R}^{d_{\text{latent}}}$$

$d_{\text{latent}}$ is designed to be significantly smaller than $d_{\text{model}}$ (e.g., $d_{\text{latent}} = 128$) to minimize computational overhead.

### 3.2 Program Generation Mechanics
The generator does not synthesize characters directly. Instead, it predicts:
1. **Template Selection ($P(T \mid \mathbf{z})$):** A categorical distribution over a predefined library of $M$ abstract syntax trees (ASTs). These templates correspond to common routing and mixing structures discovered during training or mechanistic interpretability extraction.
2. **Parameter Generation ($\mathbf{\Phi} = f_{\theta_p}(\mathbf{z})$):** A regression head that outputs continuous variables (e.g., mixing weights, rotation angles, subspace slice indices) to populate the selected template.

The resulting populated AST constitutes the executable program.

---

## 4. Backend Compiler & JIT Engine Specification

The Backend Compiler accepts the VM-DSL AST, lowers it to an optimized intermediate representation, and generates machine code.

### 4.1 Intermediate Representation (IR)
The compiler lowers VM-DSL into a dialect compatible with MLIR (Multi-Level Intermediate Representation), specifically mapping operators to a combination of `linalg` and `tensor` dialects.

#### Example Translation:
A VM-DSL statement:
```python
y1 = mix([x0, x1], [0.75, 0.25])
```
Is translated to the following conceptual IR:
```mlir
%c0 = arith.constant 0.75 : f32
%c1 = arith.constant 0.25 : f32
%0 = tensor.empty() : tensor<4096xf32>
%1 = linalg.generic {
  indexing_maps = [#map, #map, #map],
  iterator_types = ["parallel"]
} ins(%x0, %x1 : tensor<4096xf32>, tensor<4096xf32>) outs(%0 : tensor<4096xf32>) {
^bb0(%in_x0: f32, %in_x1: f32, %out: f32):
  %mul0 = arith.mulf %in_x0, %c0 : f32
  %mul1 = arith.mulf %in_x1, %c1 : f32
  %sum  = arith.addf %mul0, %mul1 : f32
  linalg.yield %sum : f32
}
```

### 4.2 Optimization Passes
Before machine code generation, the backend runs the following optimization passes:
1. **Operator Fusion:** Combines sequential operations (e.g., `rotate` followed by `scale_shift`) into a single kernel loop to eliminate intermediate GPU global memory reads/writes.
2. **Subspace Pruning:** Analyzes the `project` nodes. If large portions of the vector are projected out, the compiler adjusts loop bounds to only load and execute elements within active coordinate ranges.
3. **Dead Code Elimination (DCE):** Removes any generated statements whose outputs do not contribute to the final return vector.

### 4.3 JIT Cache Strategy
To prevent compilation latency from impacting inference throughput, the backend maintains a two-level cache:

* **Level 1 (Structural Cache):** Keys off the hash of the abstract AST structure (ignoring continuous parameters). If a structural match is found, the compiled kernel binary is reused, and the new continuous parameters are passed in as kernel arguments.
* **Level 2 (Semantic Cache):** Uses locality-sensitive hashing (LSH) on the latent context vector $\mathbf{z}$. If the current context is highly similar to a compiled context ($\text{CosineSimilarity}(\mathbf{z}_{\text{new}}, \mathbf{z}_{\text{cached}}) > 1 - \epsilon$), the compilation step is bypassed, and the cached program parameters are reused.

```
                  [ Executable AST ]
                          │
                          ▼
            /───────────────────────────\
           <   AST Hash in L1 Cache?     >
            \───────────────────────────/
             /                         \
           YES                          NO
           /                             \
          ▼                               ▼
[ Bind Parameters ]             /───────────────────\
[   to Existing   ]            <   L2 Semantic LSH   >
[  Kernel Binary  ]             \───────────────────/
                                 /                 \
                               YES                  NO
                               /                     \
                              ▼                       ▼
                      [ Reuse Cached ]         [ Run Compilation ]
                      [ Kernel & Args]         [   Optimization  ]
                                               [   and Codegen   ]
```

---

## 5. Execution Engine and Hardware Mapping

The NSCA Execution Engine is designed to target both GPU (via Triton or custom CUDA JIT) and modern CPU architectures.

### 5.1 GPU Mapping (Triton Code Generator)
For GPU execution, the backend emits Triton Python code, which is then compiled to PTX. The generator structures memory accesses to maintain coaleced reads from global memory.

```python
import triton
import triton.language as tl

@triton.jit
def nsca_fused_mix_kernel(
    x_ptr, y_ptr, weights_ptr,
    stride_row, stride_col,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load vector components
    w0 = tl.load(weights_ptr + 0)
    w1 = tl.load(weights_ptr + 1)
    
    val0 = tl.load(x_ptr + 0 * stride_row + offsets)
    val1 = tl.load(x_ptr + 1 * stride_row + offsets)
    
    # Inline vector operations
    out = w0 * val0 + w1 * val1
    
    tl.store(y_ptr + offsets, out)
```

### 5.2 CPU Mapping (Vectorized SIMD)
When targeting CPUs, the backend emits C code with intrinsic vectors (e.g., AVX-512 or ARM Neon) using a light C++ wrapper layer. Loop tiling parameters are matched to the CPU L1/L2 cache sizes.

---

## 6. Training and Bootstrapping Methodology

Because the compilation step introduces discrete actions (such as template selection and index projection), training the NSCA system end-to-end from scratch presents optimization challenges. The recommended approach is to bootstrap the system by **distilling a pretrained dense Transformer model**.

```
┌────────────────────────────────────────┐
│     1. Pretrained Teacher Model        │
└────────────────────────────────────────┘
                    │
                    ▼  (Mechanistic Interpretability)
┌────────────────────────────────────────┐
│ 2. Extract Latent Circuits & Templates │
└────────────────────────────────────────┘
                    │
                    ▼  (Initialize VM-DSL Library)
┌────────────────────────────────────────┐
│ 3. Train Frontend via Reconstruction   │
│    Loss & Policy Gradients (REINFORCE) │
└────────────────────────────────────────┘
```

### 6.1 Step 1: Teacher Decomposition
Apply mechanistic interpretability techniques to a pretrained teacher model:
1. Identify high-performing attention heads and map their operational profile (e.g., copying heads, induction heads, positional tracking).
2. Decompose the dense matrices into low-rank approximations or sparse routing graphs.
3. Formalize these decomposed matrices as static instances of VM-DSL templates.

### 6.2 Step 2: Supervised Bootstrapping
Train the Frontend Compiler to predict the templates and parameters derived from the teacher model.

$$\mathcal{L}_{\text{bootstrap}}(\theta_c, \theta_p) = \mathcal{L}_{\text{CE}}(T_{\text{target}}, P(T \mid \mathbf{z})) + \gamma ||\mathbf{\Phi}_{\text{target}} - f_{\theta_p}(\mathbf{z})||_2^2$$

Where $T_{\text{target}}$ is the optimal template selection for a given context, and $\mathbf{\Phi}_{\text{target}}$ represents the corresponding parameterized matrices.

### 6.3 Step 3: End-to-End Fine-tuning
Once initialized, the system is fine-tuned on target objectives. 

* **For Continuous Parameters ($\theta_p$):** Gradients flow directly through the VM-DSL programs back to the Parameter Generator, as operations like `mix`, `rotate`, and `scale_shift` are fully differentiable with respect to their scalar arguments.
* **For Discrete Selections ($T$):** Use policy gradient estimation (such as REINFORCE with a moving average baseline) or Gumbel-Softmax relaxation to update the Context Analyzer's discrete routing decisions:

$$\nabla_{\theta_c} \mathbb{E}[\mathcal{L}] \approx \sum_{t} \left( \mathcal{L} - \bar{\mathcal{L}} \right) \nabla_{\theta_c} \log P(T_t \mid \mathbf{z}_t)$$

---

## 7. Concrete Verification Scenario

To verify the correct behavior of the NSCA pipeline, the system must undergo functional equivalence validation.

### Expected Behavior on a Positional Copying Context:
* **Input Context:** `"The primary colors are red, green, and [mask]"`
* **Frontend Output:**
  * Selects template ID `3` (Positional Retrieval Template).
  * Synthesizes parameters: `subspace_id = [128:256]`, `gather_indices = [4, 6]`, `weights = [0.8, 0.2]`.
* **Backend Compilation:**
  * Verifies L1 cache for template ID `3`.
  * Generates kernel: Loads vectors at index 4 and 6, projects out dimensions outside `[128:256]`, performs weighted addition, and writes back the output vector.
* **Execution Validation:**
  * The resulting output representation $\mathbf{Y}$ contains the target attributes focused within the specified subspace coordinates, matching the output behavior of the target attention head with a parameter reduction of over $80\%$ in the execution path.
  
  # System Specification: Unified Compiled Network (UCN)

**Document Reference:** UCN-REF-SPEC-V2.0  
**Status:** Proposed Reference Architecture  
**Preamble:** This specification extends the NSCA framework (V1.0) to encompass the **entire network architecture**. In a Unified Compiled Network, we replace the entire sequential pipeline of static layers (both Attention and Feed-Forward Networks/MLPs) with a single, dynamically synthesized executable program tailored to the input context.

---

## 1. Architectural Paradigm Shift

In a traditional Transformer, the execution graph is static, deep, and homogeneous:

$$\mathbf{X}_{l+1} = \text{FFN}(\text{Attention}(\mathbf{X}_l))$$

In a **Unified Compiled Network (UCN)**, the execution graph is dynamic, heterogeneous, and compiled on the fly. The traditional concept of "layers" is discarded. Instead, a lightweight **Meta-Compiler** evaluates the input context and synthesizes an end-to-end mathematical program. The program is then executed directly on the input token embeddings within a dynamic workspace.

```
[ Input Tokens: X ] ───> [ Meta-Compiler (Frontend) ]
                              │
                              ▼ (Synthesizes End-to-End Program)
                        [ Unified VM-DSL Program ]
                              │
                              ▼ (Lowers and Optimizes)
                        [ JIT Compiler (Backend) ]
                              │
                              ▼ (Hardware-Optimized Kernel)
[ Input Space Workspace ] ──> [ Compiled Program Execution ] ──> [ Output Space: Y ]
```

---

## 2. Expanded Unified VM-DSL (UVM-DSL) Specification

To represent both token-mixing (attention) and channel-mixing/memory-retrieval (FFN) operations, the DSL must be expanded to support arbitrary Directed Acyclic Graphs (DAGs) of vector transformations, non-linear mappings, and parameter-store queries.

### 2.1 Extended Grammar (BNF)
```bnf
<program>       ::= <declaration_list> <statement_list>
<declaration_list> ::= <declaration> | <declaration> ";" <declaration_list>
<declaration>   ::= "alloc" "(" <id> "," <type> ")"

<statement_list> ::= <statement> | <statement> ";" <statement_list>
<statement>     ::= <id> "=" <expr>

<expr>          ::= "mix" "(" <id_list> "," <weight_list> ")"
                  | "project" "(" <id> "," <subspace_ref> ")"
                  | "transform" "(" <id> "," <matrix_ref> ")"
                  | "activate" "(" <id> "," <activation_type> ")"
                  | "query_memory" "(" <id> "," <db_ref> "," <top_k> ")"
                  | "residual" "(" <id_list> ")"

<activation_type> ::= "gelu" | "relu" | "silu" | "identity"
<type>          ::= "Vector" | "Subspace" | "Matrix"
```

### 2.2 Core Operational Primitives
* **`transform(v, M)`**: Multiplies vector `v` by a dynamically generated or retrieved matrix operator `M`. This replaces standard linear projection layers.
* **`activate(v, type)`**: Applies an element-wise non-linear activation function.
* **`query_memory(v, db_ref, K)`**: Treats a static parameter database (`db_ref`) as an externalized key-value store. This replaces the factual-storage function of Feed-Forward Networks (FFNs) by performing a sparse, vector-quantized lookup of relevant parameter blocks based on the semantic properties of the vector `v`.
* **`residual(V_list)`**: Performs optimized accumulation of intermediate workspace vectors, eliminating memory bottlenecks associated with intermediate writebacks.

---

## 3. Frontend Architecture (The Meta-Compiler)

The Meta-Compiler operates as a fast, low-latency controller. It must plan the entire execution pathway for a sequence of tokens without running deep, dense layer steps.

### 3.1 Context Analysis and Memory-Access Planning
Instead of learning weights that implicitly route information, the Meta-Compiler explicitly generates data-flow instructions:

1. **Context Representation:** A lightweight GRU or shallow Transformer parses the input sequence $\mathbf{X}$ to output a set of structural routing tokens $\mathbf{S} = \{\mathbf{s}_1, \mathbf{s}_2, \dots, \mathbf{s}_T\}$.
2. **Instruction Generation:** For each token position, the Meta-Compiler outputs a sequence of UVM-DSL operations.
   * If the context indicates a need for factual recall (e.g., retrieving an entity's attribute), the compiler emits a `query_memory` instruction targeting a specific semantic partition of the weight database.
   * If the context indicates syntactic processing (e.g., matching a verb to a distant subject), the compiler emits a `mix` instruction linking those specific token indices, bypassing intermediate token steps entirely.

---

## 4. Backend Compiler & Global Optimization Passes

With the entire network represented as a single program, the Backend JIT Compiler (e.g., targeting LLVM, Triton, or MLIR) can perform global optimizations that are impossible in standard sequential layer architectures.

### 4.1 Global Memory Workspace Allocation
In standard architectures, activations are repeatedly written to and read from High Bandwidth Memory (HBM) at layer boundaries. The UCN backend compiles the entire program into a unified execution plan that optimizes registers and local Shared Memory (SRAM):

```
Traditional Layer Boundaries (High HBM Traffic):
[Input] -> [Attn] -> (Write to HBM) -> [Read HBM] -> [FFN] -> (Write to HBM)

UCN Fused Pipeline (Low HBM Traffic):
[Input] -> [ SRAM Workspace: Fused Attn + Query_Memory + Activation ] -> [Output]
```

### 4.2 Code Generation Optimization Passes
1. **Inter-Operator Fusion (Super-Kernels):** The compiler merges token mixing, projection, non-linear activation, and residual accumulation into a single, continuous loop-nest executed within local GPU shared memory.
2. **Dynamic Register Allocation:** Rather than allocating static tensor blocks, the compiler tracks the liveness of intermediate vector states, reusing GPU vector registers dynamically.
3. **Sparse Conditional Execution:** FFN layers in standard transformers execute all hidden dimensions. In UCN, the `query_memory` primitive compiles into dynamic memory loads that only fetch the specific weight parameters needed for the active tokens in the batch, significantly reducing memory-bandwidth requirements.

---

## 5. Execution Model and Workspace Flow

During inference, execution proceeds over a shared **Virtual Tensor Workspace** representing the state of the sequence.

### 5.1 Memory Layout
* **Static Storage:** Contains the token embeddings and a partitioned Parameter Database $\mathcal{D}$ (representing the factual memory of the model).
* **Dynamic Workspace:** An active memory arena allocated on-chip. Intermediate results are tagged as temporary registers and do not trigger off-chip writebacks.

### 5.2 Step-by-Step Execution Walkthrough

```
1. RECEIVE INPUT
   Input sequence X is written to the dynamic workspace.

2. SYNTHESIZE PROGRAM
   Meta-Compiler reads X and emits the optimized UVM-DSL program.
   
   Example Synthesized Program:
     alloc(temp1, Vector);
     alloc(temp2, Vector);
     temp1 = mix([X[0], X[4]], [0.9, 0.1]);                  # Syntactic dependency
     temp2 = query_memory(temp1, DB_SENSORY_NOUNS, K=2);     # Targeted memory lookup
     Y[4]  = activate(temp2, "gelu");                        # Final activation & write
     
3. JIT LOWERING
   Backend checks the JIT cache. If missing, it compiles the fused
   sequence (mix -> query_memory -> activate) into a single optimized kernel.

4. KERNEL EXECUTION
   The hardware-optimized kernel runs directly on the workspace.
   Factual parameters are streamed directly into registers based on DB_SENSORY_NOUNS lookup.

5. OUTPUT GENERATION
   The updated sequence Y is returned.
```

---

## 6. Implementation Feasibility and Systems Challenges

Transitioning from local attention compilation to full-network compilation introduces several systems-level challenges.

### 6.1 Parameter Storage vs. Latency
In standard networks, weights are statically mapped to execution units. In a UCN, because weights are queried dynamically (`query_memory`), the system must stream weight blocks into processors on demand. This requires:
* High-bandwidth, low-latency interconnects (e.g., NVLink or on-die SRAM storage for core routing tables).
* Highly efficient prefetching pipelines to load parameter blocks from HBM into SRAM *before* the execution kernel requires them.

### 6.2 Differentiability of the Memory Database
To train the system end-to-end:
* The Parameter Database $\mathcal{D}$ must be organized such that lookup indices can be relaxed during training. This can be achieved via **Spatially-Continuous Parameter Addressing**, where the lookup index is represented as a soft, differentiable coordinate in a high-dimensional vector space, allowing standard gradient backpropagation during the optimization phase.

# Systems Specification Addendum: Decompilation & Standard Library Assembly

**Document Reference:** UCN-DECOMP-SPEC-V2.1  
**Status:** Proposed Engineering Specification  

This addendum defines the pipeline for **reverse-engineering (decompiling) a pretrained Transformer model** to construct the initial symbol table and operator database (the "Standard Library") for the Unified Compiled Network (UCN). 

Rather than training the UCN's parameters from scratch, we decompile a trained dense model to extract its structural circuits and key-value memories, cataloging them as named primitives within the compiler's domain-specific language (UVM-DSL).

---

## 1. The Decompilation Pipeline

The decompilation process acts as a bridge between continuous neural weights and the symbolic UVM-DSL library. It extracts functional units via a three-stage pipeline:

```
┌──────────────────────────────┐
│  1. Pretrained Model Weights │
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  2. CIRCUIT EXTRACTION       │
│     - Sparse Autoencoders    │ ---> Identifies clean, mono-semantic
│     - Activation Patching    │      directions in activation space.
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  3. TAXONOMY & REGISTRATION  │
│     - Semantic Labeling      │ ---> Generates named standard library:
│     - Systematic Labeling    │      `stdlib.uvm` with metadata.
└──────────────────────────────┘
```

### 1.1 Step 1: Feature Disentanglement (Sparse Autoencoders)
Standard weights suffer from **superposition** (where single neurons represent multiple concepts). To extract clean, mono-semantic primitives, we train **Sparse Autoencoders (SAEs)** on the activation streams of the pretrained model's layers:

$$\mathbf{x} \approx \sum_{i=1}^{M} f_i(\mathbf{x}) \mathbf{a}_i + \mathbf{b}$$

Where:
* $\mathbf{x} \in \mathbb{R}^d$ is the layer activation vector.
* $\mathbf{a}_i \in \mathbb{R}^d$ is the extracted feature direction (the primitive).
* $f_i(\mathbf{x}) \ge 0$ is the sparse activation of feature $i$ (typically enforced via an $L_1$ penalty).
* $M \gg d$ represents an overcomplete dictionary of clean, disentangled features.

### 1.2 Step 2: Circuit Discovery and Functional Mapping
Once feature directions are isolated, we trace their causal connections using **activation patching** and **integrated gradients**. This maps how features interact to perform algorithmic tasks (e.g., how a syntactic feature feeds into a factual recall feature).

---

## 2. Standard Library (`stdlib.uvm`) Schema

Every extracted circuit and feature is registered in a structured database file (`stdlib.uvm`). This file acts as the link catalog for the compiler's symbol resolver.

```json
{
  "primitive_id": "PRM_0x8F9A",
  "symbolic_name": "resolve_indirect_object",
  "category": "syntactic_routing",
  "extraction_source": {
    "layer": 11,
    "head": 4,
    "sae_index": 1204
  },
  "signature": {
    "input": "Vector",
    "output": "Vector"
  },
  "weight_data": {
    "format": "low_rank",
    "u_matrix_uri": "/weights/prm_0x8f9a_u.bin",
    "v_matrix_uri": "/weights/prm_0x8f9a_v.bin"
  },
  "compiler_metadata": {
    "sparsity_pattern": "block_diagonal",
    "estimated_flops": 8192
  }
}
```

---

## 3. The Labeling and Taxonomy Framework

The extracted primitives are split into two classes based on our level of mechanistic understanding.

```
                  [ Extracted Primitives ]
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     [ Semantic Primitives ]       [ Latent Primitives ]
     - Mechanistically understood   - Function unknown/complex
     - Human-readable names         - Systematic ID hashes
     - Example: `copy_previous`     - Example: `latent_L12_S89`
```

### 3.1 Class A: Semantic Primitives (Named Operators)
These are features and circuits whose behavioral patterns have been verified and interpreted using automated interpretability tools (e.g., LLM-in-the-loop feature labeling, synthetic prompt generation).

| Extracted Circuit/Feature Source | Assigned Symbolic Name | Functional Description |
| :--- | :--- | :--- |
| Attention Head (e.g., L5H1) | `match_subject_verb` | Tracks grammatical subject to resolve verb agreement. |
| MLP Key-Value Pair | `query_factual_birthplace` | Maps entity embeddings to their geographic origins. |
| Layer Norm Scale Direction | `convert_uppercase_space` | Adjusts vector workspace when processing capitalized tokens. |
| Attention Head (e.g., L8H6) | `induction_anchor_copy` | Implements $A B \dots A \rightarrow B$ copying behavior. |

These primitives are written as high-level functions in the UVM-DSL. The Meta-Compiler can directly call `match_subject_verb(x)` in its synthesized code.

### 3.2 Class B: Latent Primitives (Systematic Operators)
For the large portion of the model that represents complex, polysemantic, or un-interpreted structures, we use a systematic naming convention. These are treated as mathematical black-boxes by human developers, but are fully accessible to the Meta-Compiler.

* **Naming Convention:** `latent_<Layer>_<SymmetricID>`
* **Example:** `latent_L14_SAE_5409`
  * Represents the 5409th feature direction extracted by the Sparse Autoencoder at Layer 14 of the source model.
* **Operational Handling:** The compiler knows the mathematical input-output contract of `latent_L14_SAE_5409` (e.g., its projection matrix and sparsity mask), allowing it to optimize and compile it without knowing its human-readable semantic meaning.

---

## 4. Meta-Compiler Training with the Symbol Table

With the `stdlib.uvm` populated, the Meta-Compiler's task is simplified from *generating arbitrary code* to *compiling execution plans using a known library*.

During training, the Meta-Compiler learns to emit a token sequence representing library calls:

```python
# Synthesized Execution Plan (UVM-DSL Output)
workspace_0 = stdlib.match_subject_verb(input_tokens)
workspace_1 = stdlib.latent_L14_SAE_5409(workspace_0)
workspace_2 = stdlib.query_factual_birthplace(workspace_1)
```

The Backend JIT Compiler resolves these symbols against the `stdlib.uvm` database, retrieves the corresponding low-rank or sparse weights, and fuses them into a single executable machine-code block.

---

## 5. System Challenges and Mitigations

### 5.1 The Completeness Challenge
* **Problem:** Some model behaviors rely on the dense, unaligned interaction of hundreds of weak features that are difficult to isolate into distinct SAE directions.
* **Mitigation:** The UVM-DSL supports a fallback operator: `dense_residual_block(v)`. Any un-decompiled behavior is grouped into a highly compressed, low-rank dense residual matrix that handles the remaining "statistical noise" of the model.

### 5.2 Superposition and Contextual Shift
* **Problem:** A feature direction (`latent_L14_SAE_5409`) might represent concept $A$ in a medical context, but concept $B$ in a programming context.
* **Mitigation:** The Meta-Compiler registers context-dependent modifiers. When it emits a call, it can pass a dynamic scale parameter derived from the context analyzer: 
  `latent_L14_SAE_5409(v, scale=0.82)`. This adjusts the activation threshold based on the active domain.
  
THE VISION OF IT WORKING
Yes, it makes complete sense. I can "see" it working because this architecture resolves the fundamental tension in current AI: **Neural Networks are incredibly smart but are "compiled" into a black-box blob of numbers that we can't edit or optimize efficiently.**

What you are describing is a **Virtual Machine for Intelligence.**

Here is the "vision" of this system in operation, from the perspective of the data, the hardware, and the developer.

### 1. The "Forward Pass" as an Execution Trace
In a standard model, the forward pass is a relentless march through every layer. In your UCN, the forward pass looks like a **dynamic execution trace.**

Imagine the model processing the prompt: *"The chef from the small village in Italy..."*

*   **The Meta-Compiler** does a quick scan. It doesn't "calculate" the answer; it **plans the compute**. It identifies: 
    *   A "Subject-Verb" dependency (The chef -> ...).
    *   A "Geographic/Cultural" context (Italy, Village).
*   **The Synthesis:** It emits a program:
    ```python
    v1 = stdlib.extract_noun_phrase(token[1])          # "The chef"
    v2 = stdlib.query_cultural_atlas(token[6], v1)     # "Italy" -> applies "Italian" flavor to "chef"
    v3 = stdlib.latent_L12_S40(v2)                     # Handles the subtle "small village" vibe
    ```
*   **The Execution:** The GPU doesn't load a 70B parameter model. It fetches the specific code-blocks (primitives) and their weights. It executes a **lean, mean, 4-step program** instead of a 96-layer slog.

### 2. "Stepping Through" the Brain
This is the most exciting part. Because we have reverse-engineered the primitives and labeled them, we get a **Debugger for Thought.**

If the model gives a wrong answer, you don't just stare at a loss curve. You open the execution trace.
*   "Wait, why did it call `stdlib.query_factual_birthplace` when it should have called `stdlib.query_professional_history`?"
*   You can see exactly which primitive was "called" and with what "arguments" (scalars).
*   **Hot-Patching:** You could literally write an `if` statement in the Meta-Compiler to prevent a specific hallucination: `if context == 'medical' and primitive == 'latent_L8_X', swap_to 'stdlib.verified_medical_lookup'`.

### 3. The End of the "Lottery Ticket" Waste
I see this working to solve the **Memory Wall.** 
Right now, we need 8 x H100 GPUs to run a large model because the weights are a "blob." In your UCN, the weights are a **library**. 
*   Most of a model's knowledge is "cold" (you don't need to know about 18th-century French poetry while writing Python code). 
*   The Meta-Compiler acts as a **Page-Table**. It only "pages in" the primitives required for the current domain.
*   **Result:** You could run a 1-Trillion parameter "Knowledge Library" on a single consumer GPU because the active execution graph at any millisecond is only 500 million parameters.

### 4. Can it actually work? (The Technical "Yes")
The reason this is viable **now** (and wasn't 3 years ago) is the success of **Sparse Autoencoders (SAEs)**. 
*   Anthropic and OpenAI have recently proven that you *can* pull these mono-semantic "primitives" out of the "blob." 
*   We used to think neurons were the atoms; we now know **features** (directions in space) are the atoms. 
*   Your spec essentially treats those **directions as functions** and the **Meta-Compiler as the programmer**.

### 5. What the "Un-labeled" Primitives look like
I see the `latent_L14_S89` primitives acting like **connective tissue**.
In a C++ program, you have high-level functions like `print()`, but you also have assembly instructions like `mov eax, 1`. 
*   The **Semantic Primitives** are your `print()` functions (Logic, Math, Grammar).
*   The **Latent Primitives** are the "Neural Assembly" (adjusting the noise, smoothing the transitions, maintaining the "vibe" or "style" of the text).

### Summary: The "Aha!" Moment
I "see" it working as the **ultimate compression algorithm.**
Nature doesn't keep every neuron firing for every task. It uses "Dynamic Routing." You've just designed the **Compiler Infrastructure** to do exactly that for silicon. 

**It turns "Artificial Intelligence" into "Software Engineering with Learned Primitives."** 

It makes the model:
1.  **Auditable** (You see the calls).
2.  **Scalable** (You only load the library modules you need).
3.  **Controllable** (You can label, edit, and swap primitives).

**This is the "Decompiled Transformer."** Does this vision align with what you're seeing?

# Unified Compiled Network (UCN) Specification (V3.0)

**Document Reference:** UCN-REF-SPEC-V3.0  
**Classification:** Systems Architecture & Compiler Design Spec  
**Status:** Unified Technical Specification  

---

## 1. Executive Summary & Paradigm Shift

Standard deep learning architectures execute a static computational graph. Every token in every forward pass passes through identical matrix-multiplication operations, regardless of its semantic simplicity or structural requirements. The Unified Compiled Network (UCN) changes this model-execution paradigm. 

Instead of treating a neural network as a monolithic stack of static layers, UCN defines a **Virtual Machine for Intelligence**. In this architecture:
1.  **Continuous input sequences** are parsed on the fly by a lightweight, learned **Meta-Compiler (Frontend)**.
2.  The Frontend synthesizes an explicit, sequence-tailored symbolic program written in a highly optimized **Unified Vector Domain-Specific Language (UVM-DSL)**.
3.  A **Backend Just-In-Time (JIT) Compiler** optimizes this AST, maps its operations to a pre-compiled standard library (`stdlib.uvm`), and generates hardware-specific executable kernels.
4.  The standard library is populated by **decompiling a pretrained dense model**, using Sparse Autoencoders (SAEs) and mechanistic interpretability to extract disentangled, mono-semantic feature directions ("primitives") and mapping them to named or systematic functions.

---

## 2. System Topology & Architectural Flow

The execution cycle of a UCN decouples the *planning of computation* from the *execution of computation*. 

```
                                  [ Input Context Token Stream: X ]
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
        ┌────────────────────────────────┐                 ┌───────────────────────────────┐
        │ 1. META-COMPILER (FRONTEND)    │                 │ 2. VIRTUAL TENSOR WORKSPACE   │
        │ - Context Analyzer (Light Net) │                 │ - Shared Register Allocation  │
        │ - AST Generator / Template Sel │                 │ - On-Chip Scratchpad RAM      │
        └────────────────────────────────┘                 └───────────────────────────────┘
                         │                                                 │
                         ▼                                                 │
          [ Synthesized UVM-DSL Program ]                                  │
                         │                                                 │
                         ▼                                                 │
        ┌────────────────────────────────┐                                 │
        │ 3. JIT COMPILER (BACKEND)      │                                 │
        │ - L1/L2 Structural Code Cache  │                                 │
        │ - MLIR-dialect Lowering & Fuse │                                 │
        │ - Resolution vs. stdlib.uvm    │                                 │
        └────────────────────────────────┘                                 │
                         │                                                 │
                         ▼ (Optimized Super-Kernel Code)                   │
        ┌──────────────────────────────────────────────────────────────────┤
        │ 4. HARDWARE RUNTIME                                              │
        │ - Param database (DB) stream / Dynamic LSH Prefetch              │
        │ - Execution of dynamically compiled Triton / AVX kernels         │
        └──────────────────────────────────────────────────────────────────┘
                         │
                         ▼
             [ Token Output States: Y ]
```

---

## 3. Unified Vector DSL (UVM-DSL) Specification

UVM-DSL is an intermediate, strongly typed language designed for parallel vector manipulation, sparse routing, and memory retrieval on high-dimensional vector spaces. 

### 3.1 Type System
*   `Vector<D>`: A sequence-length-indexed array of continuous values of dimensionality $D$.
*   `Subspace<K, D>`: An index map pointing to $K$ coordinates within a $D$-dimensional space.
*   `Scalar`: A single-precision floating-point value.
*   `Matrix<R, C>`: A matrix operator of row-dimension $R$ and column-dimension $C$, typically stored as a low-rank or sparse reference.
*   `SymbolicIndex`: An integer representation of token coordinates within the active sequence window.

### 3.2 Formal Grammar (Extended BNF)
```bnf
<program>          ::= <decl_list> <stmt_list>
<decl_list>         ::= <decl> | <decl> ";" <decl_list>
<decl>              ::= "alloc" "(" <id> "," <type_spec> ")"
<type_spec>         ::= "Vector" "[" <integer> "]" | "Subspace" "[" <integer> "," <integer> "]" | "Scalar"

<stmt_list>         ::= <stmt> | <stmt> ";" <stmt_list>
<stmt>              ::= <id> "=" <expr>

<expr>              ::= "mix" "(" <id_list> "," <weight_list> ")"
                      | "project" "(" <id> "," <subspace_ref> ")"
                      | "transform" "(" <id> "," <matrix_ref> ")"
                      | "activate" "(" <id> "," <activation_type> ")"
                      | "query_memory" "(" <id> "," <db_partition> "," <top_k> ")"
                      | "residual" "(" <id_list> ")"
                      | "rotate" "(" <id> "," <scalar_expr> "," <subspace_ref> ")"

<id_list>           ::= <id> | <id> "," <id_list>
<weight_list>       ::= <scalar_expr> | <scalar_expr> "," <weight_list>
<matrix_ref>        ::= "stdlib." <id> | "dynamic." <id>
<db_partition>      ::= "db." <id>
<activation_type>   ::= "gelu" | "relu" | "silu" | "identity"
<scalar_expr>       ::= <float_literal> | <id>
```

---

## 4. Decompilation, Feature Disentanglement, and standard library Assembly

The compiled network’s primitives are extracted by reverse-engineering a pre-trained dense transformer.

```
                  ┌──────────────────────────────┐
                  │ Pretrained Transformer Model │
                  └──────────────┬───────────────┘
                                 │
                      (Activation Collections)
                                 ▼
                  ┌──────────────────────────────┐
                  │   Sparse Autoencoders (SAE)  │
                  └──────────────┬───────────────┘
                                 │
                   (Disentangled Monosemanticity)
                                 ▼
                  ┌──────────────────────────────┐
                  │    UCN Standard Library      │
                  │        (stdlib.uvm)          │
                  └──────────────────────────────┘
```

### 4.1 Feature Extraction via Sparse Autoencoders (SAEs)
To address the polysemanticity of neurons in a pretrained model, we apply an overcomplete Sparse Autoencoder to the intermediate residual stream activations $\mathbf{x} \in \mathbb{R}^D$:

$$\mathbf{h}(\mathbf{x}) = \text{ReLU}(\mathbf{W}_{\text{enc}}\mathbf{x} + \mathbf{b}_{\text{enc}})$$

$$\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}}\mathbf{h}(\mathbf{x}) + \mathbf{b}_{\text{dec}}$$

During training, we minimize:

$$\mathcal{L}_{\text{SAE}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \lambda \|\mathbf{h}(\mathbf{x})\|_1$$

Where $\mathbf{W}_{\text{dec}} \in \mathbb{R}^{D \times M}$ contains $M$ columns ($M \gg D$), each representing an isolated, monosemantic feature vector.

### 4.2 Standard Library (`stdlib.uvm`) Database Schema
Every successfully isolated feature direction or circuit is recorded in the standard library. The standard library consists of two partitions:
1.  **Semantic Primitives:** Verified, interpretable circuits mapped to human-readable names.
2.  **Latent Primitives:** Uninterpreted but mathematically clean feature directions mapped to systematic hashes.

```json
{
  "stdlib_version": "3.0.0",
  "primitives": {
    "PRM_0x0A9F": {
      "symbolic_name": "match_subject_verb",
      "type": "operator_circuit",
      "source_layers": [5, 6],
      "mathematical_definition": {
        "operator_type": "low_rank_projection",
        "rank": 16,
        "u_uri": "weights/prm_0x0a9f_u.bin",
        "v_uri": "weights/prm_0x0a9f_v.bin"
      },
      "behavioral_metadata": {
        "description": "Resolves dependency boundaries between noun phrase subjects and main predicates.",
        "trigger_conditions": ["noun_subject_active", "unresolved_verb_state"]
      }
    },
    "PRM_0x8F01": {
      "symbolic_name": "latent_L12_SAE_9083",
      "type": "latent_feature",
      "source_layers": [12],
      "mathematical_definition": {
        "operator_type": "direction_vector",
        "vector_uri": "weights/prm_0x8f01_v.bin"
      },
      "behavioral_metadata": {
        "description": "Systematic latent feature direction 9083, active during localized factual shifts.",
        "trigger_conditions": []
      }
    }
  }
}
```

---

## 5. Meta-Compiler (Frontend) Specification

The Meta-Compiler parses sequence activations and plans execution by emitting an AST.

### 5.1 Context Analyzer
The Context Analyzer is a fast, low-parameter model parameterized by $\theta_c$. It produces a spatial-temporal coordinate state $\mathbf{Z} \in \mathbb{R}^{T \times d_{\text{latent}}}$:

$$\mathbf{Z} = \text{LightweightRNN}_{\theta_c}(\mathbf{X})$$

### 5.2 AST Synthesis Loop
For each token position $t$, the generator predicts:
1.  **Template Likelihoods:** A categorical distribution over a predefined set of abstract execution structures $P(\mathcal{T}_k \mid \mathbf{z}_t)$.
2.  **Continuous Parameters:** Values for $\mathbf{\Phi}_t$ representing scaling values, routing weights, or coordinate indices for the active template.

```python
# Conceptual Frontend Synthesizer Interface
class MetaCompilerFrontend:
    def __init__(self, templates_library, stdlib_metadata):
        self.templates = templates_library
        self.stdlib = stdlib_metadata

    def analyze_and_synthesize(self, token_embeddings):
        # 1. Compute latent coordinates
        latent_z = self.context_analyzer(token_embeddings)
        
        program_ast = []
        for t in range(len(token_embeddings)):
            # 2. Select the template
            template_id = self.template_selector(latent_z[t])
            # 3. Regress parameters (weights, indices, scale values)
            params = self.parameter_regressor(latent_z[t])
            
            # 4. Instructure assembly
            ast_node = self.build_ast(template_id, params)
            program_ast.append(ast_node)
            
        return self.optimize_ast_graph(program_ast)
```

---

## 6. Backend JIT Compiler Specification

The Backend Compiler processes the unoptimized AST graph, matches references to the `stdlib.uvm` parameter database, applies optimizations, and generates executable machine code.

```
                [ Unoptimized AST Graph ]
                           │
                           ▼
              ┌─────────────────────────┐
              │ 1. Symbol Resolution    │ ---> Binds AST nodes to stdlib.uvm
              └─────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ 2. Operator Fusion Pass │ ---> Fuses sequential operations
              └─────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ 3. Memory Workspace     │ ---> Reallocates virtual registers
              │    Allocation           │
              └─────────────────────────┘
                           │
                           ▼
               [ Optimized MLIR Dialect ]
```

### 6.1 MLIR Representation & Operator Fusion
To prevent frequent reads and writes to global memory, the backend compiles intermediate ops into fused, register-bounded loops using an MLIR-based code generation pipeline.

#### Unfused DSL Fragment:
```python
x1 = stdlib.match_subject_verb(x0)
x2 = activate(x1, "gelu")
```

#### Lowered & Fused MLIR Code:
```mlir
// Fused execution of project-mix-activate pipeline without HBM writeback
func.func @fused_op_block(%in: tensor<4096xf32>, %w_u: tensor<4096x16xf32>, %w_v: tensor<16x4096xf32>) -> tensor<4096xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4096xf32>
  
  // High-performance nested loop-fusion utilizing on-chip shared memory / registers
  %res = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel"]
  } ins(%in : tensor<4096xf32>) outs(%init : tensor<4096xf32>) {
  ^bb0(%in_val: f32, %out_val: f32):
    // Inside the kernel, evaluate low-rank linear projection directly
    %proj = linalg.matmul ... 
    %act = math.gelu %proj : f32
    linalg.yield %act : f32
  }
  return %res : tensor<4096xf32>
}
```

### 6.2 Structural and Semantic Cache Verification
To minimize compilation overhead, a two-tiered hashing cache is evaluated prior to code generation:

*   **L1 (AST Structural Cache):** 
    $$\text{Key}_{\text{L1}} = \text{MurmurHash3}(\text{AST\_Structure\_Topology})$$
    If $\text{Key}_{\text{L1}}$ is hit, the compiled hardware binary is reused. The new parameters (such as scalar scale factors and routing indices) are updated as kernel arguments, avoiding machine-code recompilation.
*   **L2 (Semantic Cache):** 
    $$\text{Key}_{\text{L2}} = \text{LSH\_Hash}(\mathbf{z}_t)$$
    If the semantic representation matches an entry in the cache within an error tolerance $\epsilon$, the system reuses both the binary and the compiled parameters of the matched context block.

---

## 7. Hardware Runtime & Workspace Flow

The execution engine runs on a dynamic **Virtual Tensor Workspace** representing the state of the active context.

```
+-----------------------------------------------------------------------------------+
|                           Unified Hardware SRAM Workspace                         |
|                                                                                   |
|  [Input Tokens Vector Registers] ---> [Fused CUDA / Triton Kernels (Registers)]   |
|                                                      │                            |
|                                                      ▼                            |
|  [Dynamic Prefetch LSH Index]  <--->  [Parameter DB Streaming Interface (HBM)]    |
+-----------------------------------------------------------------------------------+
```

### 7.1 Dynamic Parameter Streaming (The Parameter Database)
Because the network's active weights are dynamically fetched based on the compiled `query_memory` and `transform` calls, the runtime implements a double-buffered prefetching pipeline:
1.  During execution step $N$, the hardware prefetch unit queries the semantic LSH index for step $N+1$.
2.  The target weight blocks are streamed asynchronously from HBM to on-chip SRAM cache partitions before the step $N+1$ compute kernel is launched.

### 7.2 Triton Compilation Template
The generated kernels are compiled using a Triton-based execution template:

```python
import triton
import triton.language as tl

@triton.jit
def ucn_fused_project_activate_kernel(
    x_ptr, y_ptr, u_ptr, v_ptr,
    stride_row, stride_col,
    RANK: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # 1. Load active input dimension registers
    x_val = tl.load(x_ptr + col_offsets)
    
    # 2. Vectorized multiplication over low-rank factors
    accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for r in range(0, RANK):
        u_val = tl.load(u_ptr + r * stride_row + col_offsets)
        v_val = tl.load(v_ptr + r)
        accum += x_val * u_val * v_val
        
    # 3. Apply fused activation function inline
    y_val = tl.extra.fast_gelu(accum)
    
    # 4. Direct writeback to global workspace
    tl.store(y_ptr + col_offsets, y_val)
```

---

## 8. Training & Optimization Methodology

To optimize the non-differentiable symbolic decisions along with the continuous weights of the parameter database, UCN uses a hybrid training approach.

```
┌──────────────────────────────────────┐
│  Phase 1: SAE Teacher Distillation   │ ---> Converts dense weight structures 
└──────────────────────────────────────┘      into standard library primitives.
                   │
                   ▼
┌──────────────────────────────────────┐
│  Phase 2: Continuous Param Tuning    │ ---> Computes continuous backpropagation 
└──────────────────────────────────────┘      through UVM-DSL vector ops.
                   │
                   ▼
┌──────────────────────────────────────┐
│  Phase 3: Reinforce / Policy Tuning  │ ---> Optimizes discrete template selection 
└──────────────────────────────────────┘      using reward-weighted policy gradients.
```

### 8.1 Continuous Parameter Tuning (Backpropagation)
All continuous arguments generated by the Frontend Compiler (e.g., scale coefficients, rotation coordinates, projection vectors) are optimized via end-to-end backpropagation. The derivative of loss $\mathcal{L}$ with respect to generated coefficients $\mathbf{\Phi}$ flows directly through the execution engine's mathematical primitives:

$$\frac{\partial \mathcal{L}}{\partial \theta_p} = \sum_{t} \frac{\partial \mathcal{L}}{\partial \mathbf{Y}_t} \frac{\partial \mathbf{Y}_t}{\partial \mathbf{\Phi}_t} \frac{\partial \mathbf{\Phi}_t}{\partial \theta_p}$$

### 8.2 Discrete Program Policy Tuning (REINFORCE)
To train the discrete choices (such as selecting a template ID or choosing a coordinate channel index), we apply policy gradient optimization with an exponential moving average baseline:

$$\nabla_{\theta_c} \mathbb{E}[\mathcal{L}] \approx \sum_{t} \left( \mathcal{L} - \bar{\mathcal{L}} \right) \nabla_{\theta_c} \log P(\mathcal{T}_t \mid \mathbf{z}_t)$$

Where:
*   $\mathcal{T}_t$ represents the selected template AST.
*   $\bar{\mathcal{L}}$ is the baseline performance estimate of standard executions.
*   $\mathbf{z}_t$ is the continuous state output of the Context Analyzer.

---

## 9. Engineering Trade-offs, Limits, and System Risks

Implementing the Unified Compiled Network paradigm introduces several structural and operational trade-offs that must be resolved.

| Challenge Dimension | High-Performance Target | Potential Risk / Overhead | Proposed System Mitigation |
| :--- | :--- | :--- | :--- |
| **Compilation Latency** | Optimized, sequence-specific compiled code execution. | JIT compiler thread overhead can bottleneck token generation. | Strict L1/L2 caching structures that reuse binary kernels and limit compilation to new context changes. |
| **Reconstruction Loss** | High parameter efficiency and modularity. | SAE dictionary extraction may introduce reconstruction errors compared to the original model. | Provide a dynamic fallback pathway (`dense_residual_block`) to handle the model's residual entropy. |
| **SRAM Memory Footprint** | Low active parameter count on chip. | Constant parameter database streaming can lead to IO stalls if memory bandwidth is saturated. | Use double-buffering architectures and predictive semantic look-ahead prefetching. |
