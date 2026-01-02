#include <torch/extension.h>

// For this portfolio, we will rely on cuBLAS for the heavy lifting
// but provide the kernel that merges weights efficiently.

__global__ void add_lora_weights_kernel(
    float* __restrict__ W, 
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    int out_features, 
    int in_features, 
    int rank,
    float alpha
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = out_features * in_features;
    if (idx >= total_elements) return;

    int row = idx / in_features;
    int col = idx % in_features;

    // Compute delta = (A[row] @ B[:col]) * alpha
    // This is naive; optimized version uses tiling.
    float delta = 0.0f;
    for (int r = 0; r < rank; ++r) {
        delta += B[r * in_features + col] * A[row * rank + r]; 
    }
    
    W[idx] += delta * alpha;
}

void merge_lora(torch::Tensor W, torch::Tensor A, torch::Tensor B, float alpha) {
    int out = W.size(0);
    int in = W.size(1);
    int rank = A.size(1);
    
    int total = out * in;
    add_lora_weights_kernel<<<(total + 255)/256, 256>>>(
        W.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
        out, in, rank, alpha
    );
}