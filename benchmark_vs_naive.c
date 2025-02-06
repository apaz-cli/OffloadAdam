#include "offload_adam.h"

// gcc benchmark_vs_naive.c -lm -O3 -march=native -fno-math-errno

#define PARAM_COUNT 10000000

static double test_impl(void step_fn(AdamOptimizer* optimizer, float* volatile params, float* volatile gradients), float** out_params) {
    float* params = (float*)malloc(PARAM_COUNT * sizeof(float));
    float* gradients = (float*)malloc(PARAM_COUNT * sizeof(float));
    if (params == NULL || gradients == NULL) {
        fprintf(stderr, "Failed to allocate %zu bytes of memory for params and gradients.\n", (size_t)PARAM_COUNT * 2 * sizeof(float));
        exit(1);
    }   

    // Create some data
    for (int i = 0; i < PARAM_COUNT; i++) {
        params[i] = (float)(i + 1);
        gradients[i] = (float)(i + 1) * 0.1f * (i % 2 == 0 ? 1 : -1);
    }

    float learning_rate = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    AdamOptimizer* optimizer = adam_init(PARAM_COUNT, learning_rate, beta1, beta2, epsilon);
    
    // Time the optimization steps
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < 100; i++) {  // Increase iterations for better timing
        step_fn(optimizer, params, gradients);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    adam_free(optimizer);
    // TODO: free(optimizer); // Move this out of adam_free and change adam_init to take pointer to AdamOptimizer.
    free(gradients);
    
    *out_params = params;
    return time_taken;
}

void verify_results(float* baseline, float* test, const char* impl_name) {
    for (int i = 0; i < PARAM_COUNT; i++) {
        if (fabsf(baseline[i] - test[i]) > 1e-5f) {
            printf("Mismatch at index %d between naive and %s: %f != %f\n", 
                   i, impl_name, baseline[i], test[i]);
            exit(1);
        }
    }
    printf("Results match between naive and %s!\n", impl_name);
}

int main(void) {
    float *params_naive, *params_avx2, *params_avx512;
    double time_naive = test_impl(adam_step_naive, &params_naive);
    printf("Naive implementation: %.3f seconds\n", time_naive);

#if defined(__AVX2__)
    double time_avx2 = test_impl(adam_step_avx256, &params_avx2);
    verify_results(params_naive, params_avx2, "AVX2");
    printf("AVX2 implementation: %.3f seconds (%.2fx speedup)\n", 
           time_avx2, time_naive/time_avx2);
    free(params_avx2);
#endif

#if defined(__AVX512F__)
    double time_avx512 = test_impl(adam_step_avx512, &params_avx512);
    verify_results(params_naive, params_avx512, "AVX-512");
    printf("AVX-512 implementation: %.3f seconds (%.2fx speedup)\n", 
           time_avx512, time_naive/time_avx512);
    free(params_avx512);
#endif

    free(params_naive);
    return 0;
}
