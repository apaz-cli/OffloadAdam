#include <torch/extension.h>
#include "offload_adam.h"

// Must install ninja to build this extension
// Also install torch and numpy

AdamOptimizer* create_optimizer(torch::Tensor& grad, float lr, float beta1, float beta2, float epsilon) {
    TORCH_CHECK(grad.is_contiguous(), "grads must be contiguous");
    TORCH_CHECK(grad.dtype() == torch::kFloat32, "grads must be float32");
    int64_t param_count = grad.numel();
    puts("Creating optimizer.");
    return adam_init(param_count, lr, beta1, beta2, epsilon);
}

void destroy_optimizer(AdamOptimizer* optimizer) {
    puts("Destroying optimizer.");
    adam_free(optimizer);
}

#define STEP_CHECKS() \
    TORCH_CHECK(optimizer != nullptr, "optimizer must not be null"); \
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous"); \
    TORCH_CHECK(grads.is_contiguous(), "grads must be contiguous"); \
    TORCH_CHECK(params.dtype() == torch::kFloat32, "params must be float32"); \
    TORCH_CHECK(grads.dtype() == torch::kFloat32, "grads must be float32"); \
    TORCH_CHECK(params.sizes() == grads.sizes(), "params and grads must have same shape"); \
    TORCH_CHECK((uint64_t)params.numel() == optimizer->param_count, "parameter count mismatch");

torch::Tensor step_naive(
    AdamOptimizer* optimizer,
    torch::Tensor& params,
    torch::Tensor& grads
) {
    STEP_CHECKS()
    adam_step_naive(optimizer, params.data_ptr<float>(),  grads.data_ptr<float>());
    return params;
}

#if defined(__AVX2__)
torch::Tensor step_avx2(
    AdamOptimizer* optimizer,
    torch::Tensor& params,
    torch::Tensor& grads
) {
    STEP_CHECKS()
    adam_step_avx2(optimizer, params.data_ptr<float>(),  grads.data_ptr<float>());
    return params;
}
#endif

#if defined(__AVX512F__)
torch::Tensor step_avx512(
    AdamOptimizer* optimizer,
    torch::Tensor& params,
    torch::Tensor& grads
) {
    STEP_CHECKS()
    adam_step_avx512(optimizer, params.data_ptr<float>(),  grads.data_ptr<float>());
    return params;
}
#endif

torch::Tensor step(
    AdamOptimizer* optimizer,
    torch::Tensor& params,
    torch::Tensor& grads
) {
#if defined(__AVX512F__)
    return step_avx512(optimizer, params, grads);
#elif defined(__AVX2__)
    return step_avx2(optimizer, params, grads);
#else
    return step_naive(optimizer, params, grads);
#endif
    return params;
}

int get_simd_level(AdamOptimizer* optimizer) {
    (void)optimizer;
#if defined(__AVX512F__)
    return 512;
#elif defined(__AVX2__)
    return 256;
#else
    return 0;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<AdamOptimizer>(m, "AdamOptimizer", py::module_local())
        .def_readonly("grad", &AdamOptimizer::param_count)
        .def_readwrite("lr", &AdamOptimizer::lr)
        .def_readwrite("beta1", &AdamOptimizer::beta1)
        .def_readwrite("beta2", &AdamOptimizer::beta2)
        .def_readwrite("eps", &AdamOptimizer::eps)
        .def_readwrite("t", &AdamOptimizer::t);

    m.def("create_optimizer", &create_optimizer, "Create Adam optimizer",
          py::arg("grad"),
          py::arg("lr") = 0.001f,
          py::arg("beta1") = 0.9f,
          py::arg("beta2") = 0.999f,
          py::arg("epsilon") = 1e-8f,
          py::return_value_policy::take_ownership);
          
    m.def("destroy_optimizer", &destroy_optimizer, "Free Adam optimizer memory");
    
    m.def("step_naive", &step_naive, "Naive Adam optimizer step",
          py::arg("optimizer"),
          py::arg("params"),
          py::arg("grads"));

#if defined(__AVX512F__)
    m.def("step_avx512", &step_avx512, "AVX512 optimized Adam optimizer step",
          py::arg("optimizer"),
          py::arg("params"),
          py::arg("grads"));
#endif
    
    m.def("step", &step, "AVX512 optimized Adam optimizer step if available, else naive.",
          py::arg("optimizer"),
          py::arg("params"),
          py::arg("grads"));

    m.def("simd_level", &get_simd_level, "Get SIMD level (0=none, 256=AVX2, 512=AVX512)",
          py::arg("optimizer"));
}
