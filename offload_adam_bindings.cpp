#include <torch/extension.h>
#include "offload_adam.h"

static AdamOptimizer* get_optimizer(int64_t param_count, float lr, float beta1, float beta2, float epsilon) {
    return adam_init(param_count, lr, beta1, beta2, epsilon);
}

torch::Tensor adam_step(
    torch::Tensor& params,
    torch::Tensor& grads,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    bool use_avx512
) {
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(params.dtype() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(grads.dtype() == torch::kFloat32, "grads must be float32");
    TORCH_CHECK(params.sizes() == grads.sizes(), "params and grads must have same shape");
    
    int64_t num_params = params.numel();
    float* params_ptr = params.data_ptr<float>();
    float* grads_ptr = grads.data_ptr<float>();

    static AdamOptimizer* optimizer = nullptr;
    if (optimizer == nullptr) {
        optimizer = get_optimizer(num_params, lr, beta1, beta2, epsilon);
    }

    if (use_avx512) {
        adam_step_avx512(optimizer, params_ptr, grads_ptr);
    } else {
        adam_step_naive(optimizer, params_ptr, grads_ptr);
    }

    return params;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam_step", &adam_step, "Optimized Adam optimizer step",
          py::arg("params"),
          py::arg("grads"),
          py::arg("learning_rate") = 0.001f,
          py::arg("beta1") = 0.9f,
          py::arg("beta2") = 0.999f,
          py::arg("epsilon") = 1e-8f,
          py::arg("use_avx512") = true);
}
