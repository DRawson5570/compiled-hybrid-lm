import numpy as np
from toy_stdlib import TOY_STDLIB


class ASTNode:
    def __init__(self, op, target, args):
        self.op = op
        self.target = target
        self.args = args


class ToyMetaCompiler:
    def __init__(self):
        pass

    def analyze_and_synthesize(self, X):
        mean_activation = np.mean(X)

        program = []
        if mean_activation > 0.1:
            program.append(ASTNode(
                op="ROTATE_SUB",
                target="workspace",
                args={"theta": 0.785}
            ))
            program.append(ASTNode(
                op="LOW_RANK_FFN",
                target="output",
                args={"scale": 1.2}
            ))
        else:
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
        c_code = """#include <stdio.h>
#include <math.h>

"""
        ffn_weights = self.stdlib["PRM_LOW_RANK_FFN"]

        c_code += "const float W_down[2][8] = {\n"
        for row in ffn_weights["W_down"]:
            c_code += "    {" + ", ".join(map(str, row)) + "},\n"
        c_code += "};\n\n"

        c_code += "const float W_up[8][2] = {\n"
        for row in ffn_weights["W_up"]:
            c_code += "    {" + ", ".join(map(str, row)) + "},\n"
        c_code += "};\n\n"

        c_code += """void pipeline_execute(const float input[8], float output[8]) {
    float workspace[8] = {0.0f};
    for(int i = 0; i < 8; i++) { workspace[i] = input[i]; }

"""

        for step, node in enumerate(ast):
            c_code += f"    // Step {step}: {node.op}\n"
            if node.op == "ROTATE_SUB":
                theta = node.args["theta"]
                c_code += f"""    {{
        float cos_t = {np.cos(theta)}f;
        float sin_t = {np.sin(theta)}f;
        float temp0 = workspace[0];
        float temp1 = workspace[1];
        workspace[0] = temp0 * cos_t - temp1 * sin_t;
        workspace[1] = temp0 * sin_t + temp1 * cos_t;
    }}

"""

            elif node.op == "LOW_RANK_FFN":
                scale = node.args["scale"]
                c_code += f"""    {{
        float bottleneck[2] = {{0.0f}};
        for (int r = 0; r < 2; r++) {{
            for (int c = 0; c < 8; c++) {{
                bottleneck[r] += workspace[c] * W_down[r][c];
            }}
        }}
        for (int r = 0; r < 2; r++) {{
            if (bottleneck[r] < 0.0f) bottleneck[r] = 0.0f;
        }}
        for (int r = 0; r < 8; r++) {{
            float accum = 0.0f;
            for (int c = 0; c < 2; c++) {{
                accum += bottleneck[c] * W_up[r][c];
            }}
            {node.target}[r] = accum * {scale}f;
        }}
    }}

"""

        c_code += """}

int main() {
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


if __name__ == "__main__":
    X = np.array([
        [0.5, -0.2, 0.8, 0.1, -0.4, 0.9, -0.1, 0.3],
        [0.1,  0.2, 0.3, 0.4,  0.5, 0.6,  0.7, 0.8]
    ], dtype=np.float32)

    frontend = ToyMetaCompiler()
    backend = ToyBackendJIT(TOY_STDLIB)

    print("Step 1: Frontend Context Analysis...")
    ast = frontend.analyze_and_synthesize(X)

    print("Step 2: Backend lowering and C-codegen...")
    backend.compile_to_c(ast)
