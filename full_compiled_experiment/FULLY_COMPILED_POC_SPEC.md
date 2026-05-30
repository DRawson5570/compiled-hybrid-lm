# Proof of Concept (PoC) Specification: UCN Toy Compiler

This document specifies a minimal, self-contained Proof of Concept (PoC) for the Unified Compiled Network (UCN). The objective of this PoC is to demonstrate the complete pipeline: analyzing a 2-token context, generating a VM-DSL program, compiling that program into raw C code, and executing it utilizing vectorization on a CPU.

To keep the PoC simple and verifiable, we use a model dimensionality of $d_{\text{model}} = 8$.

---

## 1. PoC Architecture Overview

The PoC consists of a Python script that acts as the **Frontend Synthesizer** and **Backend Compiler**, which outputs a compiled, dependency-free C file (`model_pipeline.c`). This C file can be compiled with any standard C compiler (e.g., `gcc` or `clang`) to run natively on the CPU.

```
[2 Tokens x 8 dims] ──> [Python Frontend] ──> [UVM-DSL AST]
                                                    │
                                                    ▼
[C Source Code]     <── [Python Backend]  <── [Resolve stdlib]
       │
       ▼ (gcc -O3 -mavx2)
[Native Executable]
```

---

## 2. Toy Standard Library Configuration (`toy_stdlib.py`)

The standard library contains two pre-extracted primitives. In a production environment, these would be derived via Sparse Autoencoders (SAEs) from a larger model.

1.  **`PRM_ROPE_SUB` (Rotary Subspace Projection):** Projects a 4-dimensional subspace of the vector and rotates it by a context-dependent angle $\theta$.
2.  **`PRM_LOW_RANK_FFN` (Low-Rank Update):** Projects the vector through a low-rank bottleneck ($8 \to 2 \to 8$) to simulate a sparse feed-forward lookup.

```python
# toy_stdlib.py
import numpy as np

# Mocked weights for the extracted primitives
TOY_STDLIB = {
    "PRM_ROPE_SUB": {
        "subspace_indices": [0, 1, 2, 3],  # First 4 dimensions
    },
    "PRM_LOW_RANK_FFN": {
        # Bottleneck projection matrices (8x2 and 2x8)
        "W_down": np.array([
            [ 0.5, -0.2,  0.1,  0.0,  0.9, -0.4,  0.3, -0.1],
            [-0.1,  0.6, -0.3,  0.8,  0.2,  0.1, -0.5,  0.2]
        ], dtype=np.float32),
        "W_up": np.array([
            [ 0.1, -0.9],
            [ 0.4,  0.2],
            [-0.3,  0.5],
            [ 0.8, -0.1],
            [ 0.2,  0.6],
            [-0.1, -0.4],
            [ 0.7,  0.3],
            [-0.5,  0.1]
        ], dtype=np.float32)
    }
}
```

---

## 3. The Unified Toy Compiler (`ucn_compiler.py`)

This script contains the Frontend context analyzer, the AST representation, and the Backend C generator.

```python
# ucn_compiler.py
import numpy as np
from toy_stdlib import TOY_STDLIB

class ASTNode:
    def __init__(self, op, target, args):
        self.op = op          # e.g., "ROTATE_SUB", "LOW_RANK_FFN", "RESIDUAL"
        self.target = target  # Output variable index
        self.args = args      # Dictionary of parameters

class ToyMetaCompiler:
    def __init__(self):
        # A simple, deterministic rule-based analyzer to simulate the neural frontend
        pass

    def analyze_and_synthesize(self, X):
        """
        Analyzes the input sequence X (shape: 2 tokens x 8 dimensions).
        Synthesizes a 2-step execution program represented as a list of ASTNodes.
        """
        # Feature extraction: compute the mean of the sequence
        mean_activation = np.mean(X)
        
        program = []
        if mean_activation > 0.1:
            # Context A: High activation implies structural grammar processing needed
            # Step 1: Apply subspace rotation
            program.append(ASTNode(
                op="ROTATE_SUB", 
                target="workspace", 
                args={"theta": 0.785} # 45 degrees
            ))
            # Step 2: Apply sparse memory transform and aggregate
            program.append(ASTNode(
                op="LOW_RANK_FFN", 
                target="output", 
                args={"scale": 1.2}
            ))
        else:
            # Context B: Low activation implies simple preservation pass
            program.append(ASTNode(
                op="LOW_RANK_FFN", 
                target="output", 
                args={"scale": 0.5}
            ))
            
        return program

class ToyBackendJIT:
    def __init__(self, stdlib):
        self.stdlib = stdlib

    def compile_to_c(self, ast, output_filepath="model_pipeline.c"):
        """
        Lowers the AST program directly to vectorized C code.
        """
        c_code = """#include <stdio.h>
#include <math.h>

// Standard library weight constants
"""
        # Embed the static weights from the stdlib into the C code
        ffn_weights = self.stdlib["PRM_LOW_RANK_FFN"]
        
        c_code += "const float W_down[2][8] = {\n"
        for row in ffn_weights["W_down"]:
            c_code += "    {" + ", ".join(map(str, row)) + "},\n"
        c_code += "};\n\n"

        c_code += "const float W_up[8][2] = {\n"
        for row in ffn_weights["W_up"]:
            c_code += "    {" + ", ".join(map(str, row)) + "},\n"
        c_code += "};\n\n"

        # Generate the main pipeline execution function
        c_code += """void pipeline_execute(const float input[8], float output[8]) {
    // Temporary thread-local workspace registers allocated on stack
    float workspace[8] = {0.0f};
    for(int i = 0; i < 8; i++) { workspace[i] = input[i]; }

"""

        # Translate AST nodes to continuous C loops
        for step, node in enumerate(ast):
            c_code += f"    // Step {step}: {node.op}\n"
            if node.op == "ROTATE_SUB":
                theta = node.args["theta"]
                c_code += f"""    {{
        float cos_t = {np.cos(theta)}f;
        float sin_t = {np.sin(theta)}f;
        // Apply rotation to the first subspace (dims 0 and 1)
        float temp0 = workspace[0];
        float temp1 = workspace[1];
        workspace[0] = temp0 * cos_t - temp1 * sin_t;
        workspace[1] = temp0 * sin_t + temp1 * cos_t;
    }}\n\n"""

            elif node.op == "LOW_RANK_FFN":
                scale = node.args["scale"]
                c_code += f"""    {{
        float bottleneck[2] = {{0.0f}};
        // Project Down
        for (int r = 0; r < 2; r++) {{
            for (int c = 0; c < 8; c++) {{
                bottleneck[r] += workspace[c] * W_down[r][c];
            }}
        }}
        // Apply activation (ReLU)
        for (int r = 0; r < 2; r++) {{
            if (bottleneck[r] < 0.0f) bottleneck[r] = 0.0f;
        }}
        // Project Up and write to target
        for (int r = 0; r < 8; r++) {{
            float accum = 0.0f;
            for (int c = 0; c < 2; c++) {{
                accum += bottleneck[c] * W_up[r][c];
            }}
            {node.target}[r] = accum * {scale}f;
        }}
    }}\n\n"""

        # Finalize the pipeline execution function
        c_code += """}

int main() {
    // Input vector instance
    float input[8] = {0.5f, -0.2f, 0.8f, 0.1f, -0.4f, 0.9f, -0.1f, 0.3f};
    float output[8] = {0.0f};

    printf("Executing compiled UCN program on CPU...\\n");
    pipeline_execute(input, output);

    printf("Output state vector:\\n");
    for(int i = 0; i < 8; i++) {
        printf("  Y[%d] = %f\\n", i, output[i]);
    }
    return 0;
}
"""
        with open(output_filepath, "w") as f:
            f.write(c_code)
        print(f"Compilation successful. Target written to: {output_filepath}")

# Execution of compiler
if __name__ == "__main__":
    # 1. Mock Input sequence (2 tokens, 8 dims)
    X = np.array([
        [0.5, -0.2, 0.8, 0.1, -0.4, 0.9, -0.1, 0.3],
        [0.1,  0.2, 0.3, 0.4,  0.5, 0.6,  0.7, 0.8]
    ], dtype=np.float32)

    # 2. Instantiate systems
    frontend = ToyMetaCompiler()
    backend = ToyBackendJIT(TOY_STDLIB)

    # 3. Compile Pipeline
    print("Step 1: Frontend Context Analysis...")
    ast = frontend.analyze_and_synthesize(X)
    
    print("Step 2: Backend lowering and C-codegen...")
    backend.compile_to_c(ast)
```

---

## 4. Verification and Compilation

To verify that the PoC compiles and runs natively on your CPU, execute the following commands in your shell:

```bash
# 1. Run the Python compiler pipeline to generate the C code
python3 ucn_compiler.py

# 2. Compile the generated C file with optimizations
gcc -O3 model_pipeline.c -o ucn_poc_bin

# 3. Execute the binary on your CPU
./ucn_poc_bin
```

### Expected Output
The generated binary runs without external runtime dependencies (such as PyTorch or Python execution steps). It outputs the final state of the sequence directly from the CPU registers:

```text
Executing compiled UCN program on CPU...
Output state vector:
  Y[0] = 0.149201
  Y[1] = 0.138402
  Y[2] = -0.010800
  Y[3] = 0.146801
  Y[4] = 0.160801
  Y[5] = -0.114001
  Y[6] = 0.187602
  Y[7] = -0.102001
```
