#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.cuh"

// Define Tile Sizes
#define BLOCK_SIZE_M 64
#define BLOCK_SIZE_N 64
#define HEAD_DIM 64

// The actual CUDA Kernel
__global__ void flash_fwd_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V,
    float* __restrict__ O,
    const int N,     // Sequence Length
    const int d,     // Head Dimension
    const float scale
) {
    // Thread Indices
    int tx = threadIdx.x;
    int bx = blockIdx.x; // Batch * Heads
    int by = blockIdx.y; // Sequence Block M
    
    // Offset pointers to the correct batch/head
    int qkv_offset = bx * N * d;
    const float* q_ptr = Q + qkv_offset;
    const float* k_ptr = K + qkv_offset;
    const float* v_ptr = V + qkv_offset;
    float* o_ptr = O + qkv_offset;

    // Shared Memory for Tiles
    __shared__ float Q_tile[BLOCK_SIZE_M][HEAD_DIM];
    __shared__ float K_tile[BLOCK_SIZE_N][HEAD_DIM];
    __shared__ float V_tile[BLOCK_SIZE_N][HEAD_DIM];

    // Accumulators
    float acc[HEAD_DIM] = {0.0f};
    float l = 0.0f; // Softmax denominator (sum of exp)
    float m = -1e20f; // Softmax max value (for stability)

    // 1. Load Q Tile (Row by)
    int row = by * BLOCK_SIZE_M + tx;
    if (row < N) {
        for (int i = 0; i < d; ++i) {
            Q_tile[tx][i] = q_ptr[row * d + i];
        }
    }
    __syncthreads();

    // 2. Loop over K, V blocks (Columns)
    for (int j = 0; j < (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; ++j) {
        
        // Load K and V Tiles
        int col = j * BLOCK_SIZE_N + tx;
        if (col < N) {
            for (int i = 0; i < d; ++i) {
                K_tile[tx][i] = k_ptr[col * d + i];
                V_tile[tx][i] = v_ptr[col * d + i];
            }
        }
        __syncthreads();

        // Compute QK^T (Attention Scores)
        // Note: Simplification - Real FlashAttention keeps this inside registers
        // This is a naive shared memory implementation for structure demonstration
        if (row < N) {
            for (int k_idx = 0; k_idx < BLOCK_SIZE_N; ++k_idx) {
                int global_col = j * BLOCK_SIZE_N + k_idx;
                if (global_col >= N) continue;

                float score = 0.0f;
                for (int i = 0; i < d; ++i) {
                    score += Q_tile[tx][i] * K_tile[k_idx][i];
                }
                score *= scale;

                // Online Softmax Update
                float m_prev = m;
                m = max(m, score);
                float exp_score = __expf(score - m);
                float exp_correction = __expf(m_prev - m);
                
                l = l * exp_correction + exp_score;

                // Update Accumulator
                for (int i = 0; i < d; ++i) {
                    acc[i] = acc[i] * exp_correction + exp_score * V_tile[k_idx][i];
                }
            }
        }
        __syncthreads();
    }

    // 3. Write Output
    if (row < N) {
        for (int i = 0; i < d; ++i) {
            o_ptr[row * d + i] = acc[i] / l;
        }
    }
}

// Host Wrapper
torch::Tensor flash_attn_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, 
    float softmax_scale, float dropout_p, bool is_causal
) {
    CHECK_INPUT(q); CHECK_INPUT(k); CHECK_INPUT(v);

    int B = q.size(0); // Batch
    int H = q.size(1); // Heads
    int N = q.size(2); // Sequence Length
    int d = q.size(3); // Head Dim

    auto o = torch::zeros_like(q);

    // Grid Configuration
    // y-dimension covers the sequence length (tiling M)
    // x-dimension covers the batch * heads
    dim3 grid(B * H, (N + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    dim3 block(BLOCK_SIZE_M);
    
    // Dynamic shared memory? No, using static for now.
    
    flash_fwd_kernel<<<grid, block>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        o.data_ptr<float>(),
        N, d, softmax_scale
    );
    
    return o;
}