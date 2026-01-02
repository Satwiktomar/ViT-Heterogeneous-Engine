#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int head_dim,
    int seq_len,
    int total_tokens
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens * (head_dim / 2)) return;

    // Calculate indices
    int half_dim = head_dim / 2;
    int token_idx = idx / half_dim;
    int dim_idx = idx % half_dim;
    
    int seq_pos = token_idx % seq_len; // Simple position mapping
    
    // Indices for real and imaginary parts
    int i1 = token_idx * head_dim + dim_idx;
    int i2 = token_idx * head_dim + dim_idx + half_dim;
    
    float q_r = q[i1]; float q_i = q[i2];
    float k_r = k[i1]; float k_i = k[i2];
    
    float c = cos[seq_pos * half_dim + dim_idx];
    float s = sin[seq_pos * half_dim + dim_idx];

    // Apply Rotation
    q[i1] = q_r * c - q_i * s;
    q[i2] = q_r * s + q_i * c;
    
    k[i1] = k_r * c - k_i * s;
    k[i2] = k_r * s + k_i * c;
}

void apply_rope(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin) {
    int B = q.size(0);
    int H = q.size(1);
    int N = q.size(2);
    int D = q.size(3);
    
    int total_pairs = B * H * N * (D / 2);
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    
    rope_kernel<<<blocks, threads>>>(
        q.data_ptr<float>(), k.data_ptr<float>(), 
        cos.data_ptr<float>(), sin.data_ptr<float>(), 
        D, N, B*H*N
    );
}