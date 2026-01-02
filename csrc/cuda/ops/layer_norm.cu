#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void layer_norm_forward_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int N,
    int D,
    float eps
) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    // 1. Compute Mean
    float local_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local_sum += (float)x[bid * D + i];
    }

    // Warp Reduction for Sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // Broadcast sum to all threads (simplified, assumes 1 block per row)
    __shared__ float s_mean;
    if (tid == 0) s_mean = local_sum / D;
    __syncthreads();

    // 2. Compute Variance
    float local_var = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = (float)x[bid * D + i];
        local_var += (val - s_mean) * (val - s_mean);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_var += __shfl_down_sync(0xffffffff, local_var, offset);
    }

    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(local_var / D + eps);
    __syncthreads();

    // 3. Normalize and Write
    for (int i = tid; i < D; i += blockDim.x) {
        float val = (float)x[bid * D + i];
        float norm = (val - s_mean) * s_rstd;
        y[bid * D + i] = (T)(norm * (float)weight[i] + (float)bias[i]);
    }
    
    if (tid == 0) {
        mean[bid] = s_mean;
        rstd[bid] = s_rstd;
    }
}

// Host Wrapper
void layer_norm_fwd(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, 
    torch::Tensor y, torch::Tensor mean, torch::Tensor rstd, float eps
) {
    int N = x.size(0); // Batch size
    int D = x.size(1); // Hidden Dim
    
    dim3 grid(N);
    dim3 block(std::min(1024, D)); // Simple block config
    
    layer_norm_forward_kernel<float><<<grid, block>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        y.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
        N, D, eps
    );
}