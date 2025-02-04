#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// If we need to change the grad or optimizer state dtype, we shall rewrite.

typedef struct {
    volatile float* m;
    volatile float* v;
    float beta1;
    float beta2;
    float learning_rate;
    float epsilon;
    uint64_t param_count;
    uint64_t t;
} AdamOptimizer;

// Initialize the Adam optimizer
AdamOptimizer* adam_init(int param_count, float learning_rate, float beta1, float beta2, float epsilon) {
    AdamOptimizer* optimizer = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    
    optimizer->m = (float*)calloc(param_count, sizeof(float));
    optimizer->v = (float*)calloc(param_count, sizeof(float));
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->learning_rate = learning_rate;
    optimizer->epsilon = epsilon;
    optimizer->param_count = param_count;
    optimizer->t = 0;
    
    return optimizer;
}

// Free the optimizer's memory
void adam_free(AdamOptimizer* optimizer) {
    free((void*)optimizer->m);
    free((void*)optimizer->v);
    free(optimizer);
}

void adam_step(AdamOptimizer* optimizer, volatile float* params, volatile float* gradients) {
    optimizer->t += 1;
    float beta1 = powf(optimizer->beta1, optimizer->t);
    float beta2 = powf(optimizer->beta2, optimizer->t);
    float one_minus_beta1 = 1.0f - optimizer->beta1;
    float one_minus_beta2 = 1.0f - optimizer->beta2;
    float one_minus_beta1_t = 1.0f - beta1;
    float one_minus_beta2_t = 1.0f - beta2;
    
    // Process 16 elements at a time
    uint64_t i;
    for(i = 0; i + 15 < optimizer->param_count; i += 16) {
        float grad0 = gradients[i];
        float grad1 = gradients[i+1];
        float grad2 = gradients[i+2];
        float grad3 = gradients[i+3];
        float grad4 = gradients[i+4];
        float grad5 = gradients[i+5];
        float grad6 = gradients[i+6];
        float grad7 = gradients[i+7];
        float grad8 = gradients[i+8];
        float grad9 = gradients[i+9];
        float grad10 = gradients[i+10];
        float grad11 = gradients[i+11];
        float grad12 = gradients[i+12];
        float grad13 = gradients[i+13];
        float grad14 = gradients[i+14];
        float grad15 = gradients[i+15];

        float m_0 = optimizer->m[i];
        float m_1 = optimizer->m[i+1];
        float m_2 = optimizer->m[i+2];
        float m_3 = optimizer->m[i+3];
        float m_4 = optimizer->m[i+4];
        float m_5 = optimizer->m[i+5];
        float m_6 = optimizer->m[i+6];
        float m_7 = optimizer->m[i+7];
        float m_8 = optimizer->m[i+8];
        float m_9 = optimizer->m[i+9];
        float m_10 = optimizer->m[i+10];
        float m_11 = optimizer->m[i+11];
        float m_12 = optimizer->m[i+12];
        float m_13 = optimizer->m[i+13];
        float m_14 = optimizer->m[i+14];
        float m_15 = optimizer->m[i+15];

        float v_0 = optimizer->v[i];
        float v_1 = optimizer->v[i+1];
        float v_2 = optimizer->v[i+2];
        float v_3 = optimizer->v[i+3];
        float v_4 = optimizer->v[i+4];
        float v_5 = optimizer->v[i+5];
        float v_6 = optimizer->v[i+6];
        float v_7 = optimizer->v[i+7];
        float v_8 = optimizer->v[i+8];
        float v_9 = optimizer->v[i+9];
        float v_10 = optimizer->v[i+10];
        float v_11 = optimizer->v[i+11];
        float v_12 = optimizer->v[i+12];
        float v_13 = optimizer->v[i+13];
        float v_14 = optimizer->v[i+14];
        float v_15 = optimizer->v[i+15];

        float m0 = optimizer->m[i] = optimizer->beta1 * m_0 + one_minus_beta1 * grad0;
        float m1 = optimizer->m[i+1] = optimizer->beta1 * m_1 + one_minus_beta1 * grad1;
        float m2 = optimizer->m[i+2] = optimizer->beta1 * m_2 + one_minus_beta1 * grad2;
        float m3 = optimizer->m[i+3] = optimizer->beta1 * m_3 + one_minus_beta1 * grad3;
        float m4 = optimizer->m[i+4] = optimizer->beta1 * m_4 + one_minus_beta1 * grad4;
        float m5 = optimizer->m[i+5] = optimizer->beta1 * m_5 + one_minus_beta1 * grad5;
        float m6 = optimizer->m[i+6] = optimizer->beta1 * m_6 + one_minus_beta1 * grad6;
        float m7 = optimizer->m[i+7] = optimizer->beta1 * m_7 + one_minus_beta1 * grad7;
        float m8 = optimizer->m[i+8] = optimizer->beta1 * m_8 + one_minus_beta1 * grad8;
        float m9 = optimizer->m[i+9] = optimizer->beta1 * m_9 + one_minus_beta1 * grad9;
        float m10 = optimizer->m[i+10] = optimizer->beta1 * m_10 + one_minus_beta1 * grad10;
        float m11 = optimizer->m[i+11] = optimizer->beta1 * m_11 + one_minus_beta1 * grad11;
        float m12 = optimizer->m[i+12] = optimizer->beta1 * m_12 + one_minus_beta1 * grad12;
        float m13 = optimizer->m[i+13] = optimizer->beta1 * m_13 + one_minus_beta1 * grad13;
        float m14 = optimizer->m[i+14] = optimizer->beta1 * m_14 + one_minus_beta1 * grad14;
        float m15 = optimizer->m[i+15] = optimizer->beta1 * m_15 + one_minus_beta1 * grad15;

        float v0 = optimizer->v[i] = optimizer->beta2 * v_0 + one_minus_beta2 * grad0 * grad0;
        float v1 = optimizer->v[i+1] = optimizer->beta2 * v_1 + one_minus_beta2 * grad1 * grad1;
        float v2 = optimizer->v[i+2] = optimizer->beta2 * v_2 + one_minus_beta2 * grad2 * grad2;
        float v3 = optimizer->v[i+3] = optimizer->beta2 * v_3 + one_minus_beta2 * grad3 * grad3;
        float v4 = optimizer->v[i+4] = optimizer->beta2 * v_4 + one_minus_beta2 * grad4 * grad4;
        float v5 = optimizer->v[i+5] = optimizer->beta2 * v_5 + one_minus_beta2 * grad5 * grad5;
        float v6 = optimizer->v[i+6] = optimizer->beta2 * v_6 + one_minus_beta2 * grad6 * grad6;
        float v7 = optimizer->v[i+7] = optimizer->beta2 * v_7 + one_minus_beta2 * grad7 * grad7;
        float v8 = optimizer->v[i+8] = optimizer->beta2 * v_8 + one_minus_beta2 * grad8 * grad8;
        float v9 = optimizer->v[i+9] = optimizer->beta2 * v_9 + one_minus_beta2 * grad9 * grad9;
        float v10 = optimizer->v[i+10] = optimizer->beta2 * v_10 + one_minus_beta2 * grad10 * grad10;
        float v11 = optimizer->v[i+11] = optimizer->beta2 * v_11 + one_minus_beta2 * grad11 * grad11;
        float v12 = optimizer->v[i+12] = optimizer->beta2 * v_12 + one_minus_beta2 * grad12 * grad12;
        float v13 = optimizer->v[i+13] = optimizer->beta2 * v_13 + one_minus_beta2 * grad13 * grad13;
        float v14 = optimizer->v[i+14] = optimizer->beta2 * v_14 + one_minus_beta2 * grad14 * grad14;
        float v15 = optimizer->v[i+15] = optimizer->beta2 * v_15 + one_minus_beta2 * grad15 * grad15;

        float m_hat0 = m0 / one_minus_beta1_t;
        float m_hat1 = m1 / one_minus_beta1_t;
        float m_hat2 = m2 / one_minus_beta1_t;
        float m_hat3 = m3 / one_minus_beta1_t;
        float m_hat4 = m4 / one_minus_beta1_t;
        float m_hat5 = m5 / one_minus_beta1_t;
        float m_hat6 = m6 / one_minus_beta1_t;
        float m_hat7 = m7 / one_minus_beta1_t;
        float m_hat8 = m8 / one_minus_beta1_t;
        float m_hat9 = m9 / one_minus_beta1_t;
        float m_hat10 = m10 / one_minus_beta1_t;
        float m_hat11 = m11 / one_minus_beta1_t;
        float m_hat12 = m12 / one_minus_beta1_t;
        float m_hat13 = m13 / one_minus_beta1_t;
        float m_hat14 = m14 / one_minus_beta1_t;
        float m_hat15 = m15 / one_minus_beta1_t;

        float v_hat0 = v0 / one_minus_beta2_t;
        float v_hat1 = v1 / one_minus_beta2_t;
        float v_hat2 = v2 / one_minus_beta2_t;
        float v_hat3 = v3 / one_minus_beta2_t;
        float v_hat4 = v4 / one_minus_beta2_t;
        float v_hat5 = v5 / one_minus_beta2_t;
        float v_hat6 = v6 / one_minus_beta2_t;
        float v_hat7 = v7 / one_minus_beta2_t;
        float v_hat8 = v8 / one_minus_beta2_t;
        float v_hat9 = v9 / one_minus_beta2_t;
        float v_hat10 = v10 / one_minus_beta2_t;
        float v_hat11 = v11 / one_minus_beta2_t;
        float v_hat12 = v12 / one_minus_beta2_t;
        float v_hat13 = v13 / one_minus_beta2_t;
        float v_hat14 = v14 / one_minus_beta2_t;
        float v_hat15 = v15 / one_minus_beta2_t;

        params[i] -= optimizer->learning_rate * m_hat0 / (sqrtf(v_hat0) + optimizer->epsilon);
        params[i+1] -= optimizer->learning_rate * m_hat1 / (sqrtf(v_hat1) + optimizer->epsilon);
        params[i+2] -= optimizer->learning_rate * m_hat2 / (sqrtf(v_hat2) + optimizer->epsilon);
        params[i+3] -= optimizer->learning_rate * m_hat3 / (sqrtf(v_hat3) + optimizer->epsilon);
        params[i+4] -= optimizer->learning_rate * m_hat4 / (sqrtf(v_hat4) + optimizer->epsilon);
        params[i+5] -= optimizer->learning_rate * m_hat5 / (sqrtf(v_hat5) + optimizer->epsilon);
        params[i+6] -= optimizer->learning_rate * m_hat6 / (sqrtf(v_hat6) + optimizer->epsilon);
        params[i+7] -= optimizer->learning_rate * m_hat7 / (sqrtf(v_hat7) + optimizer->epsilon);
        params[i+8] -= optimizer->learning_rate * m_hat8 / (sqrtf(v_hat8) + optimizer->epsilon);
        params[i+9] -= optimizer->learning_rate * m_hat9 / (sqrtf(v_hat9) + optimizer->epsilon);
        params[i+10] -= optimizer->learning_rate * m_hat10 / (sqrtf(v_hat10) + optimizer->epsilon);
        params[i+11] -= optimizer->learning_rate * m_hat11 / (sqrtf(v_hat11) + optimizer->epsilon);
        params[i+12] -= optimizer->learning_rate * m_hat12 / (sqrtf(v_hat12) + optimizer->epsilon);
        params[i+13] -= optimizer->learning_rate * m_hat13 / (sqrtf(v_hat13) + optimizer->epsilon);
        params[i+14] -= optimizer->learning_rate * m_hat14 / (sqrtf(v_hat14) + optimizer->epsilon);
        params[i+15] -= optimizer->learning_rate * m_hat15 / (sqrtf(v_hat15) + optimizer->epsilon);
    }
    
    // Handle remaining elements
    for(; i < optimizer->param_count; i++) {
        float grad = gradients[i];
        float m_ = optimizer->m[i];
        float v_ = optimizer->v[i];

        float m = optimizer->m[i] = optimizer->beta1 * m_ + one_minus_beta1 * grad;
        float v = optimizer->v[i] = optimizer->beta2 * v_ + one_minus_beta2 * grad * grad;

        float m_hat = m / one_minus_beta1_t;
        float v_hat = v / one_minus_beta2_t;
        params[i] -= optimizer->learning_rate * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
    }
}


int param_count = 41;

float* test_impl() {

    static float params[] = {0};
    float gradients[] = {0};

    for (int i = 0; i < param_count; i++) {
        params[i] = (float)(i + 1);
        gradients[i] = (float)(i + 1) * 0.1f * (i % 2 == 0 ? 1 : -1);
    }

    float learning_rate = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    AdamOptimizer* optimizer = adam_init(param_count, learning_rate, beta1, beta2, epsilon);

    // Perform one optimization step
    printf("Before optimization:\n");
    for(int i = 0; i < param_count; i++) {
        printf("param[%d] = %f\n", i, params[i]);
    }
    
    for (int i = 0; i < 3; i++) {
        adam_step(optimizer, params, gradients);
    }

    printf("\nAfter optimization:\n");
    for(int i = 0; i < param_count; i++) {
        printf("param[%d] = %f\n", i, params[i]);
    }
    
    // Free memory
    adam_free(optimizer);
    
    return params;
}

// Example usage
int main() {
    test_impl();
    return 0;
}
