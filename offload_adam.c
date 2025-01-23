#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float* mv;
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
    
    optimizer->mv = (float*)calloc(param_count * 2, sizeof(float));
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
    free(optimizer->mv);
    free(optimizer);
}

void adam_step(AdamOptimizer* optimizer, float* params, float* gradients) {
    optimizer->t += 1;
    float beta1 = powf(optimizer->beta1, optimizer->t);
    float beta2 = powf(optimizer->beta2, optimizer->t);
    
    for(int i = 0; i < optimizer->param_count; i++) {
        int idx = i * 2;
        int m_ = optimizer->mv[idx];     // m_t-1
        int v_ = optimizer->mv[idx + 1]; // v_t-1

        // Calculate m_t
        int m = optimizer->mv[idx] = (optimizer->beta1 * m_) + (1.0f - optimizer->beta1) * gradients[i];
        
        // Calculate v_t
        int v = optimizer->mv[idx + 1] = (optimizer->beta2 * v_) + (1.0f - optimizer->beta2) * gradients[i] * gradients[i];

        // Calculate parameter update
        float m_hat = m / (1.0f - beta1);
        float v_hat = v / (1.0f - beta2);
        params[i] -= optimizer->learning_rate * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
    }
}

#if __has_include(<immintrin.h>) && defined(__AVX512F__)
#include <immintrin.h>

void adam_step_avx512(AdamOptimizer* optimizer, float* params, float* gradients) {
    optimizer->t += 1;
    const __m512 beta1_t = _mm512_set1_ps(powf(optimizer->beta1, optimizer->t));
    const __m512 beta2_t = _mm512_set1_ps(powf(optimizer->beta2, optimizer->t));
    const __m512 beta1 = _mm512_set1_ps(optimizer->beta1);
    const __m512 beta2 = _mm512_set1_ps(optimizer->beta2);
    const __m512 one_minus_beta1 = _mm512_set1_ps(1.0f - optimizer->beta1);
    const __m512 one_minus_beta2 = _mm512_set1_ps(1.0f - optimizer->beta2);
    const __m512 lr = _mm512_set1_ps(optimizer->learning_rate);
    const __m512 eps = _mm512_set1_ps(optimizer->epsilon);
    
    // Process 16 parameters at a time
    int vec_size = optimizer->param_count / 16;
    for(int i = 0; i < vec_size; i++) {
        int base_idx = i * 16;
        
        // Load 16 parameters and gradients
        __m512 grad = _mm512_loadu_ps(&gradients[base_idx]);
        __m512 param = _mm512_loadu_ps(&params[base_idx]);
        
        // Load m and v vectors
        __m512 m = _mm512_loadu_ps(&optimizer->mv[base_idx * 2]);
        __m512 v = _mm512_loadu_ps(&optimizer->mv[base_idx * 2 + 16]);
        
        // Update biased first moment estimate
        m = _mm512_add_ps(
            _mm512_mul_ps(beta1, m),
            _mm512_mul_ps(one_minus_beta1, grad)
        );
        
        // Update biased second raw moment estimate
        v = _mm512_add_ps(
            _mm512_mul_ps(beta2, v),
            _mm512_mul_ps(one_minus_beta2, _mm512_mul_ps(grad, grad))
        );
        
        // Compute bias-corrected moment estimates
        __m512 m_hat = _mm512_div_ps(m, _mm512_sub_ps(_mm512_set1_ps(1.0f), beta1_t));
        __m512 v_hat = _mm512_div_ps(v, _mm512_sub_ps(_mm512_set1_ps(1.0f), beta2_t));
        
        // Update parameters
        param = _mm512_sub_ps(
            param,
            _mm512_div_ps(
                _mm512_mul_ps(lr, m_hat),
                _mm512_add_ps(_mm512_sqrt_ps(v_hat), eps)
            )
        );
        
        // Store results
        _mm512_storeu_ps(&params[base_idx], param);
        _mm512_storeu_ps(&optimizer->mv[base_idx * 2], m);
        _mm512_storeu_ps(&optimizer->mv[base_idx * 2 + 16], v);
    }
    
    // Handle remaining parameters
    for(int i = vec_size * 16; i < optimizer->param_count; i++) {
        int idx = i * 2;
        float m_ = optimizer->mv[idx];
        float v_ = optimizer->mv[idx + 1];
        
        float m = optimizer->mv[idx] = optimizer->beta1 * m_ + (1.0f - optimizer->beta1) * gradients[i];
        float v = optimizer->mv[idx + 1] = optimizer->beta2 * v_ + (1.0f - optimizer->beta2) * gradients[i] * gradients[i];
        
        float m_hat = m / (1.0f - powf(optimizer->beta1, optimizer->t));
        float v_hat = v / (1.0f - powf(optimizer->beta2, optimizer->t));
        params[i] -= optimizer->learning_rate * m_hat / (sqrtf(v_hat) + optimizer->epsilon);
    }
}
#endif

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

float* test_impl_512() {

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
        adam_step_avx512(optimizer, params, gradients);
    }
    
    // Free memory
    adam_free(optimizer);
    
    return params;
}
// Example usage
int main() {

    float* result_scalar = test_impl();
    float* result_avx512 = test_impl_512();

    for (int i = 0; i < param_count; i++) {
        printf("result_scalar[%d] = %f\n", i, result_scalar[i]);
        printf("result_avx512[%d] = %f\n", i, result_avx512[i]);
    }

}
