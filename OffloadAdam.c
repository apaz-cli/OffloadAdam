#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    double* m;           // First moment vector
    double* v;           // Second moment vector
    double beta1;        // Exponential decay rate for first moment
    double beta2;        // Exponential decay rate for second moment
    double learning_rate;
    double epsilon;      // Small constant for numerical stability
    int param_count;     // Number of parameters
    int t;              // Time step
} AdamOptimizer;

// Initialize the Adam optimizer
AdamOptimizer* adam_init(int param_count, double learning_rate, double beta1, double beta2, double epsilon) {
    AdamOptimizer* optimizer = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    
    optimizer->m = (double*)calloc(param_count, sizeof(double));
    optimizer->v = (double*)calloc(param_count, sizeof(double));
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
    free(optimizer->m);
    free(optimizer->v);
    free(optimizer);
}

// Update parameters using Adam optimization
void adam_step(AdamOptimizer* optimizer, double* params, double* gradients) {
    optimizer->t += 1;
    double beta1_t = pow(optimizer->beta1, optimizer->t);
    double beta2_t = pow(optimizer->beta2, optimizer->t);
    
    for(int i = 0; i < optimizer->param_count; i++) {
        // Update biased first moment estimate
        optimizer->m[i] = optimizer->beta1 * optimizer->m[i] + 
                         (1.0 - optimizer->beta1) * gradients[i];
        
        // Update biased second raw moment estimate
        optimizer->v[i] = optimizer->beta2 * optimizer->v[i] + 
                         (1.0 - optimizer->beta2) * gradients[i] * gradients[i];
        
        // Compute bias-corrected first moment estimate
        double m_hat = optimizer->m[i] / (1.0 - beta1_t);
        
        // Compute bias-corrected second raw moment estimate
        double v_hat = optimizer->v[i] / (1.0 - beta2_t);
        
        // Update parameters
        params[i] -= optimizer->learning_rate * m_hat / 
                    (sqrt(v_hat) + optimizer->epsilon);
    }
}

// Example usage
int main() {
    // Initialize optimizer with example values
    int param_count = 3;
    double learning_rate = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    
    AdamOptimizer* optimizer = adam_init(param_count, learning_rate, 
                                       beta1, beta2, epsilon);
    
    // Example parameters and gradients
    double params[] = {1.0, 2.0, 3.0};
    double gradients[] = {0.1, -0.2, 0.3};
    
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