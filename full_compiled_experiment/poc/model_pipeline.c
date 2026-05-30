#include <stdio.h>
#include <math.h>

const float W_down[2][8] = {
    {0.5, -0.2, 0.1, 0.0, 0.9, -0.4, 0.3, -0.1},
    {-0.1, 0.6, -0.3, 0.8, 0.2, 0.1, -0.5, 0.2},
};

const float W_up[8][2] = {
    {0.1, -0.9},
    {0.4, 0.2},
    {-0.3, 0.5},
    {0.8, -0.1},
    {0.2, 0.6},
    {-0.1, -0.4},
    {0.7, 0.3},
    {-0.5, 0.1},
};

void pipeline_execute(const float input[8], float output[8]) {
    float workspace[8] = {0.0f};
    for(int i = 0; i < 8; i++) { workspace[i] = input[i]; }

    // Step 0: ROTATE_SUB
    {
        float cos_t = 0.7073882691671998f;
        float sin_t = 0.706825181105366f;
        float temp0 = workspace[0];
        float temp1 = workspace[1];
        workspace[0] = temp0 * cos_t - temp1 * sin_t;
        workspace[1] = temp0 * sin_t + temp1 * cos_t;
    }

    // Step 1: LOW_RANK_FFN
    {
        float bottleneck[2] = {0.0f};
        for (int r = 0; r < 2; r++) {
            for (int c = 0; c < 8; c++) {
                bottleneck[r] += workspace[c] * W_down[r][c];
            }
        }
        for (int r = 0; r < 2; r++) {
            if (bottleneck[r] < 0.0f) bottleneck[r] = 0.0f;
        }
        for (int r = 0; r < 8; r++) {
            float accum = 0.0f;
            for (int c = 0; c < 2; c++) {
                accum += bottleneck[c] * W_up[r][c];
            }
            output[r] = accum * 1.2f;
        }
    }

}

int main() {
    float input[8] = {0.5f, -0.2f, 0.8f, 0.1f, -0.4f, 0.9f, -0.1f, 0.3f};
    float output[8] = {0.0f};

    printf("Executing compiled UCN program on CPU...\n");
    pipeline_execute(input, output);

    printf("Output state vector:\n");
    for(int i = 0; i < 8; i++) {
        printf("  Y[%d] = %f\n", i, output[i]);
    }
    return 0;
}
