#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#ifndef C_SEM
#define C_SEM 16
#endif

__device__ __constant__ float c_min_pos[3] = {-40.0, -40.0, -1.0};
__device__ __constant__ float c_inv_voxel  = 2.5;
__device__ __constant__ int32_t c_W = 200;
__device__ __constant__ int32_t c_H = 200;
__device__ __constant__ int32_t c_Z = 16;

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fmin(fmax(x, lo), hi);
}

__device__ __forceinline__ float sgnf(float x) {
    return (x > 0.0f) - (x < 0.0f);
}
