#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "include/common.cuh"

__global__ void sqocc_render_kernel(
    const float* __restrict__ ray_xyz,
    const float* __restrict__ ray_t,

    int32_t R,
    int32_t N,
    int32_t V,

    const int32_t* __restrict__ voxel_offsets,
    const int32_t* __restrict__ voxel_counts,
    const int32_t* __restrict__ voxel_sq_ids,

    const float* __restrict__ means,     // [Q,3]
    const float* __restrict__ rots,      // [Q,9]
    const float* __restrict__ scale,     // [Q,3]
    const float* __restrict__ eps1,      // [Q]
    const float* __restrict__ eps2,      // [Q]
    const float* __restrict__ opacity,   // [Q]
    const float* __restrict__ sem,       // [Q,C]

    float* __restrict__ out_sem,         // [R,C]
    float* __restrict__ out_depth       // [R]
) {
  int32_t r = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;

  // Accumulators per ray
  float sem_acc[C_SEM];
#pragma unroll
  for (int c = 0; c < C_SEM; ++c) sem_acc[c] = 0.0;

  float depth_acc = 0.0f;

  // Transmittance 
  float T = 1.0f;
  const float eps = 1e-10f;

  // Iterate samples along ray
  for (int32_t j = 0; j < N; ++j) {
    const int32_t idx_rn = r * N + j;

    // Sample position
    const float px = ray_xyz[(idx_rn * 3) + 0];
    const float py = ray_xyz[(idx_rn * 3) + 1];
    const float pz = ray_xyz[(idx_rn * 3) + 2];

    // Compute voxel coordinates (floor)
    int ix = __float2int_rd((px - c_min_pos[0]) * c_inv_voxel);
    int iy = __float2int_rd((py - c_min_pos[1]) * c_inv_voxel);
    int iz = __float2int_rd((pz - c_min_pos[2]) * c_inv_voxel);

    if (ix < 0 || ix >= c_W ||
        iy < 0 || iy >= c_H ||
        iz < 0 || iz >= c_Z) {
        continue;  // NOT return — continue ray marching
    }

    // Compute flat voxel id
    const int32_t vox = ix + c_W * (iy + c_H * iz);

    const int32_t count = voxel_counts[vox];
    if (count <= 0) continue;

    const int32_t start = voxel_offsets[vox];

    // Per-sample accumulators from superquadrics in voxel
    float sem_s[C_SEM];
#pragma unroll
    for (int c = 0; c < C_SEM; ++c) sem_s[c] = 0.0;

    float alpha_s = 0.0f; // will clamp to [0,1]

    // Loop superquadrics in this voxel
    for (int32_t kk = 0; kk < count; ++kk) {
      const int32_t q = voxel_sq_ids[start + kk];

      const float dx = px - means[q * 3 + 0];
      const float dy = py - means[q * 3 + 1];
      const float dz = pz - means[q * 3 + 2];

      const float* __restrict__ Rm = rots + q * 9;
      const float x = dx * Rm[0] + dy * Rm[1] + dz * Rm[2];
      const float y = dx * Rm[3] + dy * Rm[4] + dz * Rm[5];
      const float z = dx * Rm[6] + dy * Rm[7] + dz * Rm[8];

      const float inv_a = 1.0 / scale[q * 3 + 0];
      const float inv_b = 1.0 / scale[q * 3 + 1];
      const float inv_c = 1.0 / scale[q * 3 + 2];

      const float e1 = eps1[q];
      const float e2 = eps2[q];

      const float exp_xy = clampf(2.0f / e2, -1e9f, 6.0f);
      const float exp_z = clampf(2.0f / e1, -1e9f, 6.0f);
      const float exp_outer = clampf(e2 / e1, -1e9f, 6.0f);

      // bases
      const float base_x = clampf(fabsf(x * inv_a), 0.0f, 50.0f);
      const float base_y = clampf(fabsf(y * inv_b), 0.0f, 50.0f);
      const float base_z = clampf(fabsf(z * inv_c), 0.0f, 50.0f);

      // superquad kernel
      const float xy_inner = __powf(base_x, exp_xy) + __powf(base_y, exp_xy);
      const float xy_term = __powf(xy_inner, exp_outer);
      const float z_term = __powf(base_z, exp_z);

      const float contrib = __expf(-(xy_term + z_term));

      // sem + opacity
      const float* __restrict__ sem_row = sem + q * C_SEM;
#pragma unroll
      for (int cidx = 0; cidx < C_SEM; ++cidx) {
        sem_s[cidx] += contrib * sem_row[cidx];
      }
      alpha_s += contrib * opacity[q];
    }

    // Assumes alpha is density
    const float alpha = clampf(alpha_s, 0.0f, 1.0f);

    // weight = T * alpha
    const float w = T * alpha;

    // depth uses sample t
    const float tj = ray_t[j];
    depth_acc += w * tj;

    // sem uses same weights
#pragma unroll
    for (int c = 0; c < C_SEM; ++c) {
      sem_acc[c] += w * sem_s[c];
    }

    // update transmittance (exclusive)
    T *= ((1.0f + eps) - alpha);
  }

  // write outputs
  out_depth[r] = depth_acc;
#pragma unroll
  for (int c = 0; c < C_SEM; ++c) {
    out_sem[r * C_SEM + c] = sem_acc[c];
  }
}

std::vector<torch::Tensor> sqocc_render_forward_cuda(
    torch::Tensor ray_xyz,      // [R,N,3] float32
    torch::Tensor ray_t,        // [R,N] float32

    torch::Tensor voxel_offsets,   // [V] int32
    torch::Tensor voxel_counts,    // [V] int32
    torch::Tensor voxel_sq_ids,    // [T] int32

    torch::Tensor means,        // [Q,3] float32
    torch::Tensor rots,         // [Q,3,3] float32
    torch::Tensor scale,        // [Q,3] float32
    torch::Tensor eps1,         // [Q] float32
    torch::Tensor eps2,         // [Q] float32
    torch::Tensor opacity,      // [Q] float32
    torch::Tensor sem_pred    // [Q,C] float32 (C=16)
) {
  CHECK_INPUT(ray_xyz);
  CHECK_INPUT(ray_t);
  CHECK_INPUT(voxel_offsets);
  CHECK_INPUT(voxel_counts);
  CHECK_INPUT(voxel_sq_ids);
  CHECK_INPUT(means);
  CHECK_INPUT(rots);
  CHECK_INPUT(scale);
  CHECK_INPUT(sem_pred);
  CHECK_INPUT(opacity);

  TORCH_CHECK(ray_xyz.scalar_type() == at::kFloat, "ray_xyz must be float32");
  TORCH_CHECK(ray_t.scalar_type() == at::kFloat, "ray_t must be float32");
  TORCH_CHECK(voxel_offsets.scalar_type() == at::kInt, "voxel_offsets must be int32");
  TORCH_CHECK(voxel_counts.scalar_type() == at::kInt, "voxel_counts must be int32");
  TORCH_CHECK(voxel_sq_ids.scalar_type() == at::kInt, "voxel_sq_ids must be int32");

  TORCH_CHECK(ray_xyz.dim() == 3 && ray_xyz.size(2) == 3, "ray_xyz must be [R,N,3]");
  TORCH_CHECK(ray_t.dim() == 1, "ray_t must be [R,N]");

  const int32_t R = (int32_t)ray_xyz.size(0);
  const int32_t N = (int32_t)ray_xyz.size(1);
  const int32_t V = (int32_t)voxel_offsets.size(0);

  // flatten eps/opacity to 1D just in case
  auto eps1_f = eps1.contiguous().view({-1});
  auto eps2_f = eps2.contiguous().view({-1});
  auto op_f   = opacity.contiguous().view({-1});

  auto out_sem   = torch::zeros({R, C_SEM}, ray_xyz.options());
  auto out_depth = torch::zeros({R},       ray_xyz.options());

  const int threads = 256;
  const int blocks = (R + threads - 1) / threads;

  sqocc_render_kernel<<<blocks, threads>>>(
      (const float*)ray_xyz.data_ptr<float>(),
      (const float*)ray_t.data_ptr<float>(),
      R, N, V,

      (const int32_t*)voxel_offsets.data_ptr<int32_t>(),
      (const int32_t*)voxel_counts.data_ptr<int32_t>(),
      (const int32_t*)voxel_sq_ids.data_ptr<int32_t>(),

      (const float*)means.data_ptr<float>(),
      (const float*)rots.data_ptr<float>(),
      (const float*)scale.data_ptr<float>(),
      (const float*)eps1_f.data_ptr<float>(),
      (const float*)eps2_f.data_ptr<float>(),
      (const float*)op_f.data_ptr<float>(),
      (const float*)sem_pred.data_ptr<float>(),

      (float*)out_sem.data_ptr<float>(),
      (float*)out_depth.data_ptr<float>()
  );

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "sqocc_render_kernel launch failed: ", cudaGetErrorString(err));

  return {out_sem, out_depth};
}
