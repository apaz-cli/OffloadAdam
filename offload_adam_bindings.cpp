#include <torch/extension.h>
#include "offload_adam.h"

AdamOptimizer* create_optimizer(int64_t param_count, float lr, float beta1, float beta2, float epsilon) {
    return adam_init(param_count, lr, beta1, beta2, epsilon);
}

void destroy_optimizer(AdamOptimizer* optimizer) {
    adam_free(optimizer);
}

torch::Tensor step_naive(
    AdamOptimizer* optimizer,
    torch::Tensor& params,
    torch::Tensor& grads
) {
    TORCH_CHECK(optimizer != nullptr, "optimizer must not be null");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(params.dtype() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(grads.dtype() == torch::kFloat32, "grads must be float32");
    TORCH_CHECK(params.sizes() == grads.sizes(), "params and grads must have same shape");
    TORCH_CHECK(params.numel() == optimizer->param_count, "parameter count mismatch");
    
    float* params_ptr = params.data_ptr<float>();
    float* grads_ptr = grads.data_ptr<float>();
    
    adam_step_naive(optimizer, params_ptr, grads_ptr);
    return params;
}

torch::Tensor step_avx512(
    AdamOptimizer* optimizer,
    torch::Tensor& params,
    torch::Tensor& grads
) {
    TORCH_CHECK(optimizer != nullptr, "optimizer must not be null");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(params.dtype() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(grads.dtype() == torch::kFloat32, "grads must be float32");
    TORCH_CHECK(params.sizes() == grads.sizes(), "params and grads must have same shape");
    TORCH_CHECK(params.numel() == optimizer->param_count, "parameter count mismatch");
    
    float* params_ptr = params.data_ptr<float>();
    float* grads_ptr = grads.data_ptr<float>();
    
    adam_step_avx512(optimizer, params_ptr, grads_ptr);
    return params;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<AdamOptimizer>(m, "AdamOptimizer", py::module_local())
        .def_readonly("param_count", &AdamOptimizer::param_count)
        .def_readonly("learning_rate", &AdamOptimizer::learning_rate)
        .def_readonly("beta1", &AdamOptimizer::beta1)
        .def_readonly("beta2", &AdamOptimizer::beta2)
        .def_readonly("epsilon", &AdamOptimizer::epsilon)
        .def_readonly("t", &AdamOptimizer::t);

    m.def("create_optimizer", &create_optimizer, "Create Adam optimizer",
          py::arg("param_count"),
          py::arg("learning_rate") = 0.001f,
          py::arg("beta1") = 0.9f,
          py::arg("beta2") = 0.999f,
          py::arg("epsilon") = 1e-8f,
          py::return_value_policy::take_ownership);
          
    m.def("destroy_optimizer", &destroy_optimizer, "Free Adam optimizer memory");
    
    m.def("step_naive", &step_naive, "Naive Adam optimizer step",
          py::arg("optimizer"),
          py::arg("params"),
          py::arg("grads"));
          
    m.def("step_avx512", &step_avx512, "AVX512 optimized Adam optimizer step",
          py::arg("optimizer"),
          py::arg("params"),
          py::arg("grads"));
}
