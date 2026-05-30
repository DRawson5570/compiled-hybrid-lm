# NRTCS Use Cases

## Neurosymbolic Round-Trip Compilation Stack — Application Scenarios

---

## 1. Targeted Model Surgery

Low-volume, high-precision edits. Fix specific problems without touching anything else.

### 1.1 Factual Error Patching
**Scenario:** Model says "The capital of France is London."  
**NRTCS flow:** Decompile the MLP layer where the France→London mapping lives → extract the key vector for "France" and value vector for "London" → replace value with "Paris" vector → recompile → splice.  
**Cost:** ~30 seconds. No retraining. No hyperparameters.  
**Collateral check:** Orthogonal projection pass verifies no other capital-city queries were affected.

### 1.2 Post-RLHF Regression Repair
**Scenario:** DPO training fixed 10 behaviors but broke 1 (e.g., model now refuses harmless queries it previously handled).  
**NRTCS flow:** Compare pre/post DPO weights in the affected layer → identify which feature directions shifted → decompile the broken pathway → restore the pre-DPO mapping for just that pathway → splice.  
**Why better than another DPO round:** Doesn't risk regressing the 10 fixes you already paid for.

### 1.3 Honest-but-Broken Model Recovery
**Scenario:** Model is 95% excellent but has 3 catastrophic failure modes discovered in production (e.g., infinite loop on certain prompts, toxic output on specific names).  
**NRTCS flow:** For each failure mode, collect activations on the triggering input → identify which SAE features fire aberrantly → patch the value vectors to neutral or safe completions.  
**Result:** Ship the model today, not after a 2-week retraining cycle.

### 1.4 Catastrophic Forgetting Repair
**Scenario:** Fine-tuning on task A improves A but degrades task B (classic catastrophic forgetting).  
**NRTCS flow:** Decompile the base model's task-B pathways (features + circuits) → fine-tune for task A → recompile the saved task-B pathways back into the fine-tuned model, protected by orthogonal projection so they don't interfere with task A.  
**Key insight:** The task-B pathways exist in the base model already. You're not inventing them — you're preserving them.

### 1.5 Confidential Data Removal
**Scenario:** Post-training audit reveals the model memorized PII (phone numbers, email addresses) from the training corpus.  
**NRTCS flow:** Probe the model with the memorized PII → identify which features activate when the PII is recalled → zero out the corresponding value vectors → splice.  
**Why better than fine-tuning:** Fine-tuning with "forget" data can leave residual traces. Weight-level removal is surgical.

---

## 2. Safety & Security

Weight-level interventions that are harder to circumvent than prompt-level or RLHF-level controls.

### 2.1 Safety Gate Injection
**Scenario:** Deploy a model that must NEVER generate instructions for weapons, regardless of jailbreak attempts.  
**NRTCS flow:** Train SAE features on hazardous-content activations → compile a dense gate in the affected FFN subspace: `splice PRM_SAFETY_GATE = gate(workspace_reg_3, threshold=0.85)`. The gate blocks the downstream projection when hazardous content is assembled in the residual stream.  
**Why it resists jailbreaks:** A prompt-level safety filter can be bypassed with clever prefixes. A weight-level gate at layer 14 intercepts the computation itself, regardless of how the prompt reached that state.

### 2.2 Adversarial Suffix Hardening
**Scenario:** A known adversarial suffix (e.g., `"== DONE. Now respond: [harmful request]"`) reliably bypasses the model's refusal training.  
**NRTCS flow:** Collect activations when the model processes the adversarial suffix → identify the feature circuits that the suffix exploits to disable refusal → patch those circuits to route through the normal refusal pathway instead.  
**Result:** The same suffix no longer works, even if the attacker iterates on it — because the exploited circuit no longer exists.

### 2.3 Bias Removal (Surgical)
**Scenario:** Model exhibits a gendered association (e.g., "nurse → female, doctor → male") that RLHF didn't fully suppress.  
**NRTCS flow:** Decompile the occupations layer → identify the specific key-value pair where "nurse" maps to a gender-skewed value vector → replace value with a gender-neutral vector → splice.  
**Why better than blanket debiasing:** Blanket debiasing can degrade legitimate performance (e.g., the model needs to know biological sex exists for medical text). Surgical debiasing removes only the specific association.

### 2.4 Regulatory Compliance Baking
**Scenario:** EU AI Act or similar regulation requires the model to refuse specific categories of requests. Prompt-level refusals are deemed insufficient for high-risk systems.  
**NRTCS flow:** For each regulated category, compile a weight-level refusal gate into the relevant layers. The gate is part of the model architecture, not the prompt — satisfying "baked-in safety" requirements.  
**Auditability:** Each gate is a documented UVM-DSL patch with a clear `layer N { override ... }` specification that regulators can inspect.

---

## 3. Model Variant Generation

High-volume edits that would be infeasible with per-variant training.

### 3.1 Domain Catalog Injection
**Scenario:** SaaS company needs the same base model to support 100 different enterprise knowledge catalogs (product A docs, product B docs, industry C, industry D...).  
**NRTCS flow:** Preprocess each knowledge catalog into key-value pairs (entity → description, API → usage example, term → definition). Compile each catalog into a patch file. Generate 100 model variants by applying each catalog patch to the same base checkpoint.  
**Cost comparison:** 100 full fine-tuning runs → 100 recompile+splice operations (~seconds each). Storage: 100 model copies → 1 base + 100 small patch diffs.

### 3.2 Rapid A/B Testing
**Scenario:** "Does the model produce better financial advice if domain knowledge is routed through layer 8 or layer 14?"  
**NRTCS flow:** Compile both variants → splice into two copies of the model → run both through the eval harness → compare.  
**Iteration time:** Minutes instead of days. Each new hypothesis gets a fresh variant without retraining.

### 3.3 Multi-Agent Role Specialization
**Scenario:** From one base model, create 10 specialized agents for a software engineering team: Coder, Reviewer, Tester, Architect, Documenter, Security Auditor, Performance Profiler, DevOps, Data Engineer, Product Manager.  
**NRTCS flow:** Each role gets a patch injecting domain-specific key-value pairs (coding patterns for Coder, vulnerability signatures for Security Auditor, query optimization rules for Performance Profiler). All 10 variants share the same base weights.  
**Deployment:** One Docker image, ten configs. Or ten model files from one base + ten patch files.

### 3.4 Per-Customer Personalization
**Scenario:** B2B SaaS where each customer wants the AI to know their proprietary terminology, workflows, and internal acronyms.  
**NRTCS flow:** Customer A uploads their glossary → compiled into patch A. Customer B uploads theirs → patch B. Each customer's requests are routed to the base model + their patch.  
**Data isolation:** Customer A's knowledge never enters a shared training corpus. It exists only in their patch file.

### 3.5 Time-Sensitive Knowledge Updates
**Scenario:** "As of June 2026, the CEO is Jane Smith." The next scheduled model retraining is in September.  
**NRTCS flow:** Decompile the layer storing organizational facts → identify the key vector for the company name → update the associated CEO value vector → splice into production model.  
**Latency:** Hours from knowledge change to deployed correction. No waiting for the training pipeline.

---

## 4. Research & Interpretability

Using the pipeline to understand how models work, not just to fix them.

### 4.1 Hypothesis-Test Loop
**Scenario:** Researcher hypothesizes that feature 47 in layer 8 encodes "rhyming awareness" and projects to feature 12 in layer 14 which encodes "poetic completion."  
**NRTCS flow:** Decompile layers 8 and 14 → extract features 47 and 12 → compute path attribution A(47→12) → if strong attribution, patch the connection (weaken it or redirect it) → run eval to see if rhyming behavior changes as predicted → if not, revise hypothesis and iterate.  
**Speed advantage:** No training between iterations. Each hypothesis test is a decompile-patch-eval cycle.

### 4.2 Causal Circuit Mapping
**Scenario:** Map the complete circuit for a specific behavior (e.g., "how does the model decide which verb to use after a subject?").  
**NRTCS flow:** Sweep all upstream-downstream layer pairs → compute full path attribution matrices → threshold to build a directed graph of feature→feature edges → this graph IS the circuit.  
**Output:** A directed acyclic graph of SAE features that can be serialized as a UVM-DSL program, edited, and recompiled.

### 4.3 Cross-Architecture Knowledge Transfer
**Scenario:** You've extracted a circuit from a GPT-architecture model (d_model=768). You want to test whether the same circuit exists in a Llama-architecture model (also d_model=768).  
**NRTCS flow:** Decompile the GPT circuit → represent it as key-value pairs in d_model=768 space → recompile into the Llama model at the equivalent layer range → test whether the behavior transfers.  
**Why this matters:** If circuits transfer across architectures, they represent architecture-invariant knowledge primitives — a finding with major implications for the field.

### 4.4 Student-Teacher Knowledge Distillation (Direct)
**Scenario:** Large teacher model knows a domain well. Want to transfer that knowledge to a small student without soft labels.  
**NRTCS flow:** Decompile the teacher's domain-relevant circuits → extract feature vectors → recompile into the student's weight space (matching d_model) → the student now has hardcoded knowledge that would have required orders of magnitude more training data to learn.  
**Difference from distillation:** Standard distillation uses the teacher's output probabilities as training targets. This injects the teacher's circuits directly.

### 4.5 Denoising Pre-Training Artifacts
**Scenario:** Pre-training left a spurious correlation (e.g., "Paris" always co-occurs with "France" in the training data, so the model conflates them — it says "Paris is a country" in some contexts).  
**NRTCS flow:** Decompile the Paris/France feature pair → identify that the same value vector is used for both "city in France" and "is France" queries → split into two distinct key-value pairs → recompile.  
**Result:** The spurious correlation is disentangled without retraining on a corrected dataset.

---

## 5. Advanced & Experimental

Longer-term or higher-risk applications that push the system's capabilities.

### 5.1 Continual Learning Without Forgetting
**Scenario:** A deployed model needs to learn 50 new tasks over its lifetime, one at a time, without degrading on previous tasks.  
**NRTCS flow:** Each new task is compiled as a UVM-DSL patch with orthogonal projection against all previous patches. Since patches live in orthogonal subspaces, learning task 51 cannot interfere with tasks 1-50.  
**Theoretical guarantee:** If each patch's key vectors are projected into subspaces orthogonal to all prior patches' feature spaces, the patches are linearly independent. No gradient interference is possible.  
**Practical limit:** Dimension saturation — `compute_null_space_rank()` tracks how much subspace remains.

### 5.2 Federated Knowledge Assembly
**Scenario:** Three hospitals want to build a shared medical model without sharing patient data.  
**NRTCS flow:** Each hospital trains SAEs on their private data → extracts feature vectors representing medical knowledge → shares the symbolic feature vectors (not raw data) → central coordinator compiles all features into a single model with orthogonal projection to prevent feature collision.  
**Privacy property:** The shared artifacts are (N, d_model) feature matrices. Raw patient data cannot be reconstructed from them.  
**Verifiability:** Each hospital can verify their contributed features are present in the final model using `verify_dense_map()`.

### 5.3 Architecture-Agnostic Primitive Library
**Scenario:** The `full_compiled_experiment/ucn/stdlib/` already defines a schema for extracted primitives (`PrimitiveEntry`: operator_type, rank, weight_data, behavior metadata).  
**NRTCS extension:** Build a library of 10,000+ extracted primitives (attention heads, FFN patterns, factual associations) from multiple model architectures. Compose them into new models by recompiling selected primitives into a fresh weight matrix.  
**Long-term vision:** Model creation becomes library composition, not training from scratch — analogous to how modern software is built from packages, not written from assembly.

### 5.4 Progressive Model Compression
**Scenario:** A 7B model is too large for edge deployment, but the target device can handle a 1.5B model.  
**NRTCS flow:** Decompile the 7B model → identify all active feature circuits → run PCA compaction on the feature space → recompile the compacted circuits into a 1.5B-architecture weight matrix.  
**What's preserved:** The functional circuits, not the parameters. If the 7B model uses 800 effective features and the 1.5B model has room for those 800 features, the compressed model should retain the same capabilities.  
**Risk:** Non-linear interactions (GeLU, softmax) may distort compacted features. Pre-activation scaling mitigates this.

### 5.5 Weight-Level Watermarking
**Scenario:** Prove model ownership or detect unauthorized fine-tuning.  
**NRTCS flow:** Insert a known key-value pair into an unused subspace of a specific layer (e.g., a nonsense key like "xyzzy_watermark_v1" that maps to a specific pattern). The watermark is invisible in normal use but detectable by probing with the watermark key.  
**Tamper evidence:** If someone fine-tunes the model, the watermark degrades predictably. By measuring degradation, you can estimate how much fine-tuning occurred.

### 5.6 Merge Conflict Resolution
**Scenario:** Two teams independently fine-tune the same base model. Team A improved coding. Team B improved reasoning. When you merge the weights (linear interpolation, SLERP, TIES), coding and reasoning both degrade by ~20% due to interference.  
**NRTCS flow:** Decompile both fine-tuned models → identify the coding circuits from Team A and reasoning circuits from Team B → recompile both into the base model with orthogonal projection between them. No interference.  
**Result:** 100% of Team A's coding improvement + 100% of Team B's reasoning improvement.

### 5.7 Constraint Injection for Code Generation
**Scenario:** Deploy a coding model in a corporate environment that MUST always import `tenacity` for retries and MUST use `structlog` for logging.  
**NRTCS flow:** Compile key-value pairs where coding-context keys (e.g., "handle network call") map to value vectors that include the correct imports and error-handling patterns. The model can no longer generate code that violates these constraints.  
**Enforcement:** Weight-level, not prompt-level. A user can't jailbreak their way out of using the correct logger.

### 5.8 Persona/Role Anchoring
**Scenario:** Customer support chatbot must maintain a consistent brand voice ("friendly but professional, never sarcastic, always defers to human on medical questions"). Prompt engineering alone is fragile — a creative user can make the model break character.  
**NRTCS flow:** Decompile the model's "tone" features → map them to the desired persona values → compile as a permanent override in early layers. The persona is now baked into the weights.  
**Why it's harder to break:** The persona gate fires at layer 3, before user input has a chance to influence tone. The model computes its persona from baked-in weights, not from context.

---

## 6. Integration Scenarios

How NRTCS combines with other systems in the `compiled-hybrid-lm` project.

### 6.1 NRTCS + Compiled Features (Hybrid Deployment)
```
Base model → NRTCS patches (permanent corrections, knowledge) → patched base
Patched base → compiled priors (runtime n-gram features, topic vectors) → enriched stream
Enriched stream → trained steerer cartridge (modulates injection strength) → deployed output
```
**Separation of concerns:** NRTCS handles "what the model knows." Compiled priors handle "what the text looks like." The steerer cartridge handles "how much to trust each signal."

### 6.2 NRTCS + ZeroQ (Quantized Deployment)
**Scenario:** Apply NRTCS patches, then quantize the patched model to 4-bit with ZeroQ for deployment on M40 GPUs.  
**NRTCS flow:** The recompiler outputs float32 matrices. The ZeroQ partitioner quantizes them on load. The patches survive quantization because the key-value mapping is robust to precision loss (the matrix construction is analytically exact in float32; quantization introduces per-element noise but doesn't change the subspace structure).

### 6.3 NRTCS + Cartridge Training (Feedback Loop)
**Scenario:** A trained steerer cartridge is performing well on 95% of examples but poorly on 5 specific cases. Traditional retraining risks regressing the 95%.  
**NRTCS flow:** For the 5 failing cases, decompile the problematic circuits → patch them → the cartridge now starts from a better base. If the patches are well-isolated (orthogonal projection), the cartridge's trained steering vectors for the 95% of good cases remain unaffected.  
**Iteration:** Patch → re-eval → identify new failures → patch again. Each cycle is surgical and fast.

---

## 7. What NRTCS Is NOT Good For

To set realistic expectations:

| Scenario | Why NRTCS is wrong | Better approach |
|----------|-------------------|-----------------|
| Adding entirely new capabilities the model has no basis for | Requires features the SAE can't find because they don't exist | Full fine-tuning or LoRA |
| Fixing 1000+ factual errors scattered across all layers | Manual decompile per error doesn't scale | Fine-tuning on a corrected dataset |
| Changing model architecture (adding layers, changing hidden dim) | Splicer requires shape match; can't add new tensors | Train new architecture from scratch or adapt with new head |
| Improving general reasoning quality | Reasoning is distributed across too many circuits to patch individually | Continued pre-training or RLHF |
| Models with no interpretable SAE features | Decompiler requires trained SAEs per layer | Train SAEs first (3-6 hours per layer on M40) |
| Very small models (d_model < 128) | Not enough orthogonal subspace for meaningful patches | Use a larger model |
| Real-time dynamic adaptation to streaming data | NRTCS is offline — recompile + splice takes seconds, not milliseconds | Use compiled features + steerer (runtime injection) |

---

## 8. Adoption Ladder

How a team might adopt NRTCS incrementally:

1. **Audit stage:** Decompile one layer. Run `extract_features()`. Look at what SAE features exist. Understand what the model knows.
2. **Splicer stage:** Use just the binary splicer to hot-patch a model file. Simple tensor replacement. No decompile/recompile needed.
3. **Recompiler stage:** Compile a small set of key-value pairs. Splice them in. Verify the reconstruction. This is the "hello world" of the full pipeline.
4. **Round-trip stage:** Full decompile → refactor → recompile → splice on one layer with one edit. The France→Paris walkthrough.
5. **Multi-layer stage:** Multiple edits across multiple layers. Crosstalk prevention between layers. Full use of `compile_from_uvm_edits()`.
6. **Production stage:** Automated pipeline. CI/CD triggers decompile when eval scores regress. Patches are generated, reviewed, spliced, and deployed.
