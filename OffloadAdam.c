#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float* mv;          // Interleaved first and second moments [m0,v0,m1,v1,...]
    float beta1;        // Exponential decay rate for first moment
    float beta2;        // Exponential decay rate for second moment
    float learning_rate;
    float epsilon;      // Small constant for numerical stability
    int param_count;    // Number of parameters
    int t;             // Time step
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

// Update parameters using Adam optimization
void adam_step(AdamOptimizer* optimizer, float* params, float* gradients) {
    optimizer->t += 1;
    float beta1_t = powf(optimizer->beta1, optimizer->t);
    float beta2_t = powf(optimizer->beta2, optimizer->t);
    
    for(int i = 0; i < optimizer->param_count; i++) {
        int idx = i * 2;
        // Update biased first moment estimate
        optimizer->mv[idx] = optimizer->beta1 * optimizer->mv[idx] + 
                           (1.0f - optimizer->beta1) * gradients[i];
        
        // Update biased second raw moment estimate
        optimizer->mv[idx + 1] = optimizer->beta2 * optimizer->mv[idx + 1] + 
                               (1.0f - optimizer->beta2) * gradients[i] * gradients[i];
        
        // Compute bias-corrected first moment estimate
        float m_hat = optimizer->mv[idx] / (1.0f - beta1_t);
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = optimizer->mv[idx + 1] / (1.0f - beta2_t);
        
        // Update parameters
        params[i] -= optimizer->learning_rate * m_hat / 
                    (sqrtf(v_hat) + optimizer->epsilon);
    }
}

// Example usage
int main() {
    // Initialize optimizer with example values
    int param_count = 3;
    float learning_rate = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    
    AdamOptimizer* optimizer = adam_init(param_count, learning_rate, 
                                       beta1, beta2, epsilon);
    
    // Example parameters and gradients
    float params[] = {1.0f, 2.0f, 3.0f};
    float gradients[] = {0.1f, -0.2f, 0.3f};
    
    // Perform one optimization step
    printf("Before optimization:\n");
    for(int i = 0; i < param_count; i++) {
        printf("param[%d] = %f\n", i, params[i]);
    }
    
    adam_step(optimizer, params, gradients);
    
    printf("\nAfter optimization:\n");
    for(int i = 0; i < param_count; i++) {
        printf("param[%d] = %f\n", i, params[i]);
    }
    
    // Free memory
    adam_free(optimizer);
    
    return 0;
}
