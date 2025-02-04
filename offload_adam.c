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
        for(int j = 0; j < 16; j++) {
            float grad = gradients[i + j];
            float m_ = optimizer->m[i + j];
            float v_ = optimizer->v[i + j];

            float m = optimizer->m[i + j] = optimizer->beta1 * m_ + one_minus_beta1 * grad;
            float v = optimizer->v[i + j] = optimizer->beta2 * v_ + one_minus_beta2 * grad * grad;

            float m_hat = m / one_minus_beta1_t;
            float v_hat = v / one_minus_beta2_t;
            params[i + j] -= optimizer->learning_rate * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
        }
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
