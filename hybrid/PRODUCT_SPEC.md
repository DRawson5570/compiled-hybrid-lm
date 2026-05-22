THIS DOCUMENT IS DEPRECATED.
THE NEW PRODUCT SPECS ARE IN HYBRID_STRATEGY AND FRONTIER_SPEC

# PRODUCT SPECIFICATION: COMPILED MODULAR INTELLIGENCE (CMI)
### Blazing Fast, Training-Free Language Modeling on Consumer Hardware

---

## 1. Vision & Strategy

Our goal is to build an alternative to the standard, computationally expensive SGD-pretrained Transformer paradigm. We are developing a **Compiled Modular Intelligence (CMI)** architecture: a fully causal, non-parametric language model that operates at blistering speeds, scales linearly, compiles in minutes, and is trainable on simple consumer hardware (e.g., a single RTX 3080 or even a CPU).

Instead of embedding millions of concepts into dense, monolithic weight matrices through gradient-descent pretraining, we:
1. **Factor language into orthogonal, specialized structure channels (Compiled Experts)**.
2. **Compile counts, probabilities, and geometric coordinates deterministically** directly from raw corpora statistics (SGD-free).
3. **Employ ultra-lightweight, sequence-aware routing blenders** to organically dynamic-weight the specialty outputs based on lookback history.

With this approach, we aim to match the relative perplexities of modern parameterized baselines (such as GPT-2 Small, $< 29.0$ PPL) while maintaining zero-computation inference pipelines.

---

## 2. Core Operational Pillars (The 4 Expert Channels)

To deliver a frontier-competent agentic assistant, the core compiled architecture must be factored into four independent, highly specialized, SGD-free capability channels:

```
                  ┌──────────────────────────────────────────────┐
                  │            Streaming Token Stream            │
                  └──────────────┬────────┬────────┬─────────────┘
                                 │        │        │
      ┌──────────────────────────┴─┐      │        │      ┌────────────────────────────┐
      │   PPMI + SVD Space Emb     │      │        │      │ Decycled Temporal Counts   │
      └──────────────┬─────────────┘      │        │      └─────────────┬──────────────┘
                     │                    │        │                    │
                     ▼                    ▼        ▼                    ▼
        ┌────────────────────────┐  ┌──────────┐ ┌──────────┐  ┌───────────────────────┐
        │    InstructChannel     │  │ Reasoner │ │  Coder   │  │      ToolChannel      │
        │ Semantic proximity /   │  │ Multi-hop│ │ Syntax & │  │ Deterministic compute │
        │ Translation / Concepts │  │ induction│ │ Keywords │  │   & output injection  │
        └────────────┬───────────┘  └────┬─────┘ └────┬─────┘  └────────────┬──────────┘
                     │                   │            │                     │
                     ▼                   ▼            ▼                     ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │                 Sequence-Aware Blender                  │
                  │   (Lookback MLP / GRU / Dilated Causal Convolution)    │
                  └─────────────────────────┬───────────────────────────────┘
                                            ▼
                           Blended Output Probability distribution
```

### Pillar I: Instruction Following (`InstructChannel`)
*   **Role**: Handles complex, long-range semantic conceptual mappings, translations, and following conceptual guidelines.
*   **Mechanism**: Uses continuous tabular SVD space embeddings ($V=8000, d=256$) generated from a continuous Pointwise Mutual Information (PPMI) representation of the training text. It evaluates context using positional-augmented cosine similarities and performs local k-means routing ($K_{clusters}=65536$) over causal, contrastive context windows ($r_{aug} = [r_{pos}, ctx_{t} - emb_{t}]$) to find context-dependent semantic completions.

### Pillar II: Multi-Step Reasoning (`ReasonerChannel`)
*   **Role**: Triggers transitively-linked reasoning chains, multi-hop lookups, and factual deduction.
*   **Mechanism**: Implements causal multi-hop trigram and n-hop induction heads. When a pattern $A \to B \to C$ is observed in local temporal sequence caches, the channel computes the semantic and count-based probability vectors to complete the transitively closed query.

### Pillar III: Code Generation (`CoderChannel`)
*   **Role**: Predicts rigorous syntax structures, structural delimiters, keyword boundaries, and standard import boilerplate (e.g., python, numpy, imports, function declarations).
*   **Mechanism**: Tracks and maintains Compiled Symbol Signatures (syntactic bigram and trigram frequency counters specifically weighted to detect and model rare structural keyword boundaries) combined with local indent-level context variables.

### Pillar IV: Tool Use (`ToolChannel`)
*   **Role**: Detects tool-framing syntaxes, computes deterministic steps (such as math executions or databases lookups), and injects the actual computed outputs directly back into prediction distributions.
*   **Mechanism**: Detects predefined tool-invocation markers (such as trigger parameters and arguments). The channel performs real-time deterministic computation on the backend and projects the output tokens with high confidences, bypassing typical LLM hallucination and arithmetic failure modes.

---

## 3. Sequence-Aware Routing Blenders

The outputs of our various core channels are dynamically merged through a **Sequence-Aware Blender**:

$$\log P_{blend}(y \mid context_{1:t}) = \text{Blender}(context_{1:t}) \cdot \begin{pmatrix} \log P_{instruct} \\ \log P_{reason} \\ \log P_{coder} \\ \log P_{tool} \\ \log P_{ngram} \end{pmatrix}$$

We implement and support four highly advanced, sequence-aware routing architectures in `hybrid/v3_super_blender/`:

1.  **WindowMLPBlender**
    *   Fast tabular routing over a lookback window $W$.
    *   Constructs feature matrices of shape $(T, W \times F_{dim})$. Extremely lightweight and robust to state drift.
2.  **LookbackMLPBlender**
    *   A deep residual MLP (ResNet blocks with LayerNorm and GELU) processing lookback-windowed feature representations.
3.  **GRUBlender**
    *   A unidirectional sequence recurrent neural network (GRU) that causally processes raw token streams. 
    *   Utilizes chunk-by-chunk sequence forwarding and state caching to preserve context history without memory explosions.
4.  **CausalConvBlender**
    *   A 1D Causal Convolutional Network leveraging dilated convolutions with left-padded causal masks. Fully parallelizable during training and highly descriptive of sequence history.

---

## 4. Compilation & Training Guidelines (No Hacks Policy)

1.  **Strict Causality**: All context vector offsets and temporal calculations must remain strictly causal:
    $$\text{Window\_Offsets} \subset (-\infty, 0]$$
    Future target leakages (bidirectional context windows) are strictly forbidden to maintain scientific integrity.
2.  **Zero-Hack Calibration**: The sequence routing blenders must be calibrated and optimized using honest SGD training paths over pure validation splits. Code must never inspect parent stack frames or leak future target information under any evaluation mode.
3.  **Consumer Hardware Accessibility**: 
    *   Training time for the compiled database of global statistical properties and k-means clustering can be fully constructed in minutes.
    *   Inference-time blenders must rely on lightweight parameters (e.g., $< 1\text{M}$ active params), enabling CPU/laptop-level inference speeds exceeding $100$ tokens/second.

---

### Phase Targets

| Phase | Milestone | Target Metric (100K WikiText-103) | Focus |
|---|---|---|---|
| **Phase I** | Baseline Restoration | Host honest $33.0$ PPL routing | Clean implementation |
| **Phase II** | Channel Fusion | Modular integration of 4 Expert Channels | Fusion testing |
| **Phase III** | Leaderboard Breach | Break the GPT-2 boundary ($< 29.0$ PPL) | Fine-tuning & Scaling |
