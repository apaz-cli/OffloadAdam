#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

// If we need to change the grad or optimizer state dtype, we shall rewrite.

typedef struct {
    float* volatile m;
    float* volatile v;
    void* m_base;  // Original allocated pointer for m
    void* v_base;  // Original allocated pointer for v
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
    
    // Calloc and align to 64 bytes (The size of __m512).
    // This allows us to use aligned instructions.
    // Although parameters may not be aligned.
    size_t aligned_size = param_count * sizeof(float) + 63;
    optimizer->m_base = calloc(1, aligned_size);
    optimizer->v_base = calloc(1, aligned_size);
    optimizer->m = (float*)(((uintptr_t)optimizer->m_base + 63) & ~63);
    optimizer->v = (float*)(((uintptr_t)optimizer->v_base + 63) & ~63);

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
    free(optimizer->m_base);
    free(optimizer->v_base);
    free(optimizer);
}


void adam_step(AdamOptimizer* optimizer, float* volatile params, float* volatile gradients) {
    optimizer->t += 1;
    float beta1 = powf(optimizer->beta1, optimizer->t);
    float beta2 = powf(optimizer->beta2, optimizer->t);
    float one_minus_beta1 = 1.0f - optimizer->beta1;
    float one_minus_beta2 = 1.0f - optimizer->beta2;
    float one_minus_beta1_t = 1.0f - beta1;
    float one_minus_beta2_t = 1.0f - beta2;
    
    // Process 16 elements at a time using AVX-512
    uint64_t i;
    __m512 beta1_vec = _mm512_set1_ps(optimizer->beta1);
    __m512 beta2_vec = _mm512_set1_ps(optimizer->beta2);
    __m512 one_minus_beta1_vec = _mm512_set1_ps(one_minus_beta1);
    __m512 one_minus_beta2_vec = _mm512_set1_ps(one_minus_beta2);
    __m512 one_minus_beta1_t_vec = _mm512_set1_ps(one_minus_beta1_t);
    __m512 one_minus_beta2_t_vec = _mm512_set1_ps(one_minus_beta2_t);
    __m512 lr_vec = _mm512_set1_ps(optimizer->learning_rate);
    __m512 eps_vec = _mm512_set1_ps(optimizer->epsilon);

    for(i = 0; i + 15 < optimizer->param_count; i += 16) {
        // Load 16 elements
        __m512 grad_vec = _mm512_loadu_ps(&gradients[i]);
        __m512 m_prev_vec = _mm512_load_ps(&optimizer->m[i]);
        __m512 v_prev_vec = _mm512_load_ps(&optimizer->v[i]);
        __m512 param_vec = _mm512_loadu_ps(&params[i]);

        // Calculate m = beta1 * m + (1-beta1) * grad
        __m512 m_vec = _mm512_fmadd_ps(beta1_vec, m_prev_vec,
                                      _mm512_mul_ps(one_minus_beta1_vec, grad_vec));
        
        // Calculate v = beta2 * v + (1-beta2) * grad^2
        __m512 grad_sq = _mm512_mul_ps(grad_vec, grad_vec);
        __m512 v_vec = _mm512_fmadd_ps(beta2_vec, v_prev_vec,
                                      _mm512_mul_ps(one_minus_beta2_vec, grad_sq));

        // Store m and v
        _mm512_store_ps(&optimizer->m[i], m_vec);
        _mm512_store_ps(&optimizer->v[i], v_vec);

        // Calculate m_hat = m / (1-beta1^t)
        __m512 m_hat = _mm512_div_ps(m_vec, one_minus_beta1_t_vec);

        // Calculate v_hat = v / (1-beta2^t)
        __m512 v_hat = _mm512_div_ps(v_vec, one_minus_beta2_t_vec);

        // Calculate sqrt(v_hat) + epsilon
        __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), eps_vec);

        // Calculate update = lr * m_hat / (sqrt(v_hat) + eps)
        __m512 update = _mm512_div_ps(
            _mm512_mul_ps(lr_vec, m_hat),
            denom
        );

        // Update parameters
        param_vec = _mm512_sub_ps(param_vec, update);
        _mm512_storeu_ps(&params[i], param_vec);
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
