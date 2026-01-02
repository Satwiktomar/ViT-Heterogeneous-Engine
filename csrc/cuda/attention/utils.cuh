#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Helper for tiling
template<typename T>
__device__ __forceinline__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}