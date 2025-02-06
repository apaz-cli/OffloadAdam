#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// If we need to change the grad or optimizer state dtype, we shall rewrite.

typedef struct {
    struct _copyable {
        float beta1;
        float beta2;
        float lr;
        float eps;
        uint64_t param_count;
        uint64_t t;
    };
    float* volatile m; // 64-byte aligned
    float* volatile v; // 64-byte aligned
    void* m_base;      // Original allocated pointer for m
    void* v_base;      // Original allocated pointer for v
} AdamOptimizer;

// Initialize the Adam optimizer
static AdamOptimizer* adam_init(int param_count, float learning_rate, float beta1, float beta2, float eps) {
    AdamOptimizer* optimizer = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));

    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->lr = learning_rate;
    optimizer->eps = eps;
    optimizer->param_count = param_count;
    optimizer->t = 0;

    // Initialize the optimizer state.    
    // Calloc and align to 64 bytes (The size of __m512).
    // This allows us to use aligned instructions, although parameters may not be aligned.
    size_t aligned_size = param_count * sizeof(float) + 63;
    optimizer->m_base = calloc(1, aligned_size);
    optimizer->v_base = calloc(1, aligned_size);
    if (optimizer->m_base == NULL || optimizer->v_base == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes of memory for optimizer state.\n", (size_t)2 * aligned_size);
        exit(1);
    }
    optimizer->m = (float*)(((uintptr_t)optimizer->m_base + 63) & ~63);
    optimizer->v = (float*)(((uintptr_t)optimizer->v_base + 63) & ~63);
    
    return optimizer;
}

// Free the optimizer's memory
static void adam_free(AdamOptimizer* optimizer) {
    free(optimizer->m_base);
    free(optimizer->v_base);
    free(optimizer);
}

// Get total size needed for serialization
static size_t adam_get_serialized_size(const AdamOptimizer* optimizer) {
    return sizeof(struct _copyable) + 
           optimizer->param_count * sizeof(float) * 2; // m and v arrays
}

// Serialize optimizer to a pre-allocated buffer
static void adam_serialize(const AdamOptimizer* optimizer, char* buffer) {
    size_t copyable_size = sizeof(struct _copyable);
    size_t array_size = optimizer->param_count * sizeof(float);
    
    // Copy just the copyable portion
    memcpy(buffer, optimizer, copyable_size);
    
    // Copy m and v arrays
    memcpy(buffer + copyable_size, optimizer->m, array_size);
    memcpy(buffer + copyable_size + array_size, optimizer->v, array_size);
}

// Deserialize optimizer from a buffer
static AdamOptimizer* adam_deserialize(const char* buffer) {
    AdamOptimizer* optimizer = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    size_t copyable_size = sizeof(struct _copyable);
    
    // Copy just the copyable portion
    memcpy(optimizer, buffer, copyable_size);
    
    size_t header_size = sizeof(AdamOptimizer);
    size_t array_size = optimizer->param_count * sizeof(float);
    size_t aligned_size = array_size + 63;
    
    // Allocate aligned memory for m and v
    optimizer->m_base = malloc(aligned_size);
    optimizer->v_base = malloc(aligned_size);
    if (optimizer->m_base == NULL || optimizer->v_base == NULL) {
        fprintf(stderr, "Failed to allocate memory during deserialization\n");
        exit(1);
    }
    
    // Set up aligned pointers
    optimizer->m = (float*)(((uintptr_t)optimizer->m_base + 63) & ~63);
    optimizer->v = (float*)(((uintptr_t)optimizer->v_base + 63) & ~63);
    
    // Copy the arrays
    memcpy(optimizer->m, buffer + header_size, array_size);
    memcpy(optimizer->v, buffer + header_size + array_size, array_size);
    
    return optimizer;
}

static void adam_step_naive(AdamOptimizer* optimizer, float* volatile params, float* volatile gradients) {
    optimizer->t += 1;
    float beta1 = powf(optimizer->beta1, optimizer->t);
    float beta2 = powf(optimizer->beta2, optimizer->t);
    float one_minus_beta1 = 1.0f - optimizer->beta1;
    float one_minus_beta2 = 1.0f - optimizer->beta2;
    float one_minus_beta1_t = 1.0f - beta1;
    float one_minus_beta2_t = 1.0f - beta2;
    
    for(uint64_t i = 0; i < optimizer->param_count; i++) {
        float grad = gradients[i];
        float m_ = optimizer->m[i];
        float v_ = optimizer->v[i];

        float m = optimizer->m[i] = optimizer->beta1 * m_ + one_minus_beta1 * grad;
        float v = optimizer->v[i] = optimizer->beta2 * v_ + one_minus_beta2 * grad * grad;

        float m_hat = m / one_minus_beta1_t;
        float v_hat = v / one_minus_beta2_t;
        params[i] -= optimizer->lr * m_hat / (sqrtf(v_hat) + optimizer->eps);
    }
}

#if defined(__AVX2__)
#include <immintrin.h>
static void adam_step_avx256(AdamOptimizer* optimizer, float* volatile params, float* volatile gradients) {
    optimizer->t += 1;
    float beta1 = powf(optimizer->beta1, optimizer->t);
    float beta2 = powf(optimizer->beta2, optimizer->t);
    float one_minus_beta1 = 1.0f - optimizer->beta1;
    float one_minus_beta2 = 1.0f - optimizer->beta2;
    float one_minus_beta1_t = 1.0f - beta1;
    float one_minus_beta2_t = 1.0f - beta2;
    
    // Process 8 elements at a time using AVX2
    uint64_t i;
    __m256 beta1_vec = _mm256_set1_ps(optimizer->beta1);
    __m256 beta2_vec = _mm256_set1_ps(optimizer->beta2);
    __m256 one_minus_beta1_vec = _mm256_set1_ps(one_minus_beta1);
    __m256 one_minus_beta2_vec = _mm256_set1_ps(one_minus_beta2);
    __m256 one_minus_beta1_t_vec = _mm256_set1_ps(one_minus_beta1_t);
    __m256 one_minus_beta2_t_vec = _mm256_set1_ps(one_minus_beta2_t);
    __m256 lr_vec = _mm256_set1_ps(optimizer->lr);
    __m256 eps_vec = _mm256_set1_ps(optimizer->eps);

    for(i = 0; i + 7 < optimizer->param_count; i += 8) {
        // Load 8 elements
        __m256 grad_vec = _mm256_loadu_ps(&gradients[i]);
        __m256 param_vec = _mm256_loadu_ps(&params[i]);
        __m256 m_prev_vec = _mm256_load_ps(&optimizer->m[i]);
        __m256 v_prev_vec = _mm256_load_ps(&optimizer->v[i]);

        // Calculate m = beta1 * m + (1-beta1) * grad
        __m256 m_vec = _mm256_fmadd_ps(beta1_vec, m_prev_vec,
                                      _mm256_mul_ps(one_minus_beta1_vec, grad_vec));
        
        // Calculate v = beta2 * v + (1-beta2) * grad^2
        __m256 grad_sq = _mm256_mul_ps(grad_vec, grad_vec);
        __m256 v_vec = _mm256_fmadd_ps(beta2_vec, v_prev_vec,
                                      _mm256_mul_ps(one_minus_beta2_vec, grad_sq));

        // Store m and v
        _mm256_store_ps(&optimizer->m[i], m_vec);
        _mm256_store_ps(&optimizer->v[i], v_vec);

        // Calculate m_hat = m / (1-beta1^t)
        __m256 m_hat = _mm256_div_ps(m_vec, one_minus_beta1_t_vec);

        // Calculate v_hat = v / (1-beta2^t)
        __m256 v_hat = _mm256_div_ps(v_vec, one_minus_beta2_t_vec);

        // Calculate sqrt(v_hat) + eps
        __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_vec);

        // Calculate update = lr * m_hat / (sqrt(v_hat) + eps)
        __m256 update = _mm256_div_ps(
            _mm256_mul_ps(lr_vec, m_hat),
            denom
        );

        // Update parameters
        param_vec = _mm256_sub_ps(param_vec, update);
        _mm256_storeu_ps(&params[i], param_vec);
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
        params[i] -= optimizer->lr * m_hat / (sqrtf(v_hat) + optimizer->eps);
    }
}
#endif

#if defined(__AVX512F__)
#include <immintrin.h>
static void adam_step_avx512(AdamOptimizer* optimizer, float* volatile params, float* volatile gradients) {
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
    __m512 lr_vec = _mm512_set1_ps(optimizer->lr);
    __m512 eps_vec = _mm512_set1_ps(optimizer->eps);

    for(i = 0; i + 15 < optimizer->param_count; i += 16) {
        // Load 16 elements
        __m512 grad_vec = _mm512_loadu_ps(&gradients[i]);
        __m512 param_vec = _mm512_loadu_ps(&params[i]);
        __m512 m_prev_vec = _mm512_load_ps(&optimizer->m[i]);
        __m512 v_prev_vec = _mm512_load_ps(&optimizer->v[i]);

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

        // Calculate sqrt(v_hat) + eps
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
        params[i] -= optimizer->lr * m_hat / (sqrtf(v_hat) + optimizer->eps);
    }
}
#endif

