#include <torch/extension.h>

// Forward declarations of our CUDA functions
// We will implement these in the .cu files
torch::Tensor flash_attn_fwd(
    torch::Tensor q, 
    torch::Tensor k, 
    torch::Tensor v, 
    float softmax_scale,
    float dropout_p,
    bool is_causal
);

// Python Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_fwd", &flash_attn_fwd, "Flash Attention Forward (CUDA)");
}