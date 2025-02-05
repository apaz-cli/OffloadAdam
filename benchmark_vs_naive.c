#include "offload_adam.h"

// gcc benchmark_vs_naive.c -lm -O3 -march=native -fno-math-errno -mavx512f -fopt-info-vec -fsanitize=address -g -fsanitize=undefined

#if !defined(__AVX512F__)
#error "AVX-512 is required for this benchmark."
#endif


#define PARAM_COUNT 10000000

double test_impl(void step_fn(AdamOptimizer* optimizer, float* volatile params, float* volatile gradients), float** out_params) {

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
    free(gradients);
    
    *out_params = params;
    return time_taken;
}

int main(void) {
    float *params1, *params2;
    double time1 = test_impl(adam_step_naive, &params1);
    double time2 = test_impl(adam_step_avx512, &params2);

    // Verify results match
    for (int i = 0; i < PARAM_COUNT; i++) {
        if (fabsf(params1[i] - params2[i]) > 1e-5f) {
            printf("Mismatch at index %d: %f != %f\n", i, params1[i], params2[i]);
            return 1;
        }
    }

    printf("Results match!\n");
    printf("Naive implementation: %.3f seconds\n", time1);
    printf("AVX-512 implementation: %.3f seconds\n", time2);
    printf("Speedup: %.2fx\n", time1/time2);

    free(params1);
    free(params2);
    return 0;
}
