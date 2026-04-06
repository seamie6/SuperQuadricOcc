#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "include/common.cuh"

static inline __device__ float contrib_superquad(
    float x, float y, float z,
    float a, float b, float c,
    float e1, float e2)
{
  const float inv_a = 1.0 / a;
  const float inv_b = 1.0 / b;
  const float inv_c = 1.0 / c;

  const float p = clampf(2.0 / e2, -1e9, 6.0);
  const float r = clampf(2.0 / e1, -1e9, 6.0);
  const float s = clampf(e2 / e1, -1e9, 6.0);

  // bases
  const float Ax = clampf(fabsf(x * inv_a), 0.0, 50.0);
  const float Ay = clampf(fabsf(y * inv_b), 0.0, 50.0);
  const float Az = clampf(fabsf(z * inv_c), 0.0, 50.0);

  // tx = Ax^p, ty = Ay^p
  const float tx = __powf(Ax, p);
  const float ty = __powf(Ay, p);
  const float inner = tx + ty;

  const float xy = __powf(inner, s);
  const float zz = __powf(Az, r);

  return __expf(-(xy + zz));
}

// Backward kernel: one thread per ray r.
__global__ void sqocc_render_backward_kernel(
    const float* __restrict__ ray_xyz,     // [R,N,3]
    const float* __restrict__ ray_t,       // [R,N]
    int32_t R, int32_t N,

    const int32_t* __restrict__ voxel_offsets, // [V]
    const int32_t* __restrict__ voxel_counts,  // [V]
    const int32_t* __restrict__ voxel_sq_ids,  // [T]

    const float* __restrict__ means,       // [Q,3]
    const float* __restrict__ rots,        // [Q,9]
    const float* __restrict__ scale,       // [Q,3]
    const float* __restrict__ eps1,        // [Q]
    const float* __restrict__ eps2,        // [Q]
    const float* __restrict__ opacity,     // [Q]
    const float* __restrict__ sem,         // [Q,C]

    const float* __restrict__ grad_out_sem,   // [R,C]
    const float* __restrict__ grad_out_depth, // [R]

    // outputs:
    float* __restrict__ grad_means,     // [Q,3]
    float* __restrict__ grad_rots,      // [Q,9]
    float* __restrict__ grad_scale,     // [Q,3]
    float* __restrict__ grad_eps1,      // [Q]
    float* __restrict__ grad_eps2,      // [Q]
    float* __restrict__ grad_opacity,   // [Q]
    float* __restrict__ grad_sem       // [Q,C]
) {
  const int32_t r = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;

  constexpr int MAX_N = 256;
  if (N > MAX_N) return;

  // Load upstream grads for this ray
  float g_sem[C_SEM];
#pragma unroll
  for (int c = 0; c < C_SEM; ++c) {
    g_sem[c] = grad_out_sem[r * C_SEM + c];
  }
  const float g_depth = grad_out_depth[r];

  // -------- Pass 1: forward recompute per-sample alpha, T, and dot(g_sem, sem_s) --------
  float Tj[MAX_N];
  float alpha[MAX_N];
  float dot_sem_s[MAX_N]; // sum_c g_sem[c] * sem_s[c]

  // FOR OLD ALPHA CODE
  unsigned char unclamped_mask[MAX_N]; // 1 if 0<alpha_s<1

  float T = 1.0;
  const float epsT = 1e-10;

  for (int32_t j = 0; j < N; ++j) {
    Tj[j] = T;

    const int32_t idx = r * N + j;
    const float px = ray_xyz[idx * 3 + 0];
    const float py = ray_xyz[idx * 3 + 1];
    const float pz = ray_xyz[idx * 3 + 2];

    int ix = __float2int_rd((px - c_min_pos[0]) * c_inv_voxel);
    int iy = __float2int_rd((py - c_min_pos[1]) * c_inv_voxel);
    int iz = __float2int_rd((pz - c_min_pos[2]) * c_inv_voxel);

    float alpha_s = 0.0;
    float sem_s[C_SEM] = {0.0};

    if (!(ix < 0 || ix >= c_W || iy < 0 || iy >= c_H || iz < 0 || iz >= c_Z)) {
      const int32_t vox = ix + c_W * (iy + c_H * iz);
      const int32_t count = voxel_counts[vox];
      if (count > 0) {
        const int32_t start = voxel_offsets[vox];

        for (int32_t kk = 0; kk < count; ++kk) {
          const int32_t q = voxel_sq_ids[start + kk];

          const float dx = px - means[q * 3 + 0];
          const float dy = py - means[q * 3 + 1];
          const float dz = pz - means[q * 3 + 2];

          const float* Rm = rots + q * 9;
          const float x = dx * Rm[0] + dy * Rm[1] + dz * Rm[2];
          const float y = dx * Rm[3] + dy * Rm[4] + dz * Rm[5];
          const float z = dx * Rm[6] + dy * Rm[7] + dz * Rm[8];

          const float a = scale[q * 3 + 0];
          const float b = scale[q * 3 + 1];
          const float c = scale[q * 3 + 2];

          const float e1 = eps1[q];
          const float e2 = eps2[q];

          const float contrib = contrib_superquad(x, y, z, a, b, c, e1, e2);

          const float opa = opacity[q];
          alpha_s += contrib * opa;

          const float* sem_row = sem + q * C_SEM;
#pragma unroll
          for (int cc = 0; cc < C_SEM; ++cc) {
            sem_s[cc] += contrib * sem_row[cc];
          }
        }
      }
    }

    // OLD ALPHA CODE
    const float a_clamped = clampf(alpha_s, 0.0, 1.0);
    alpha[j] = a_clamped;
    unclamped_mask[j] = (alpha_s > 0.0 && alpha_s < 1.0) ? 1 : 0;

    // dot(g_sem, sem_s)
    float dotv = 0.0;
#pragma unroll
    for (int cc = 0; cc < C_SEM; ++cc) dotv += g_sem[cc] * sem_s[cc];
    dot_sem_s[j] = dotv;

    // OLD ALPHA CODE
    T *= ((1.0 + epsT) - a_clamped);
  }

  // -------- Pass 2: reverse sweep to get g_alpha_s(j) --------
  // g_w(j) = dot(g_sem, sem_s(j)) + g_depth * t(j)
  // w = T*alpha; T_{j+1} = T_j*(1-alpha)
  float gT_next = 0.0;
  float g_alpha_s[MAX_N];

  for (int32_t j = N - 1; j >= 0; --j) {
    const float gw = dot_sem_s[j] + g_depth * ray_t[j];

    // partials from w = T*alpha
    float gT = gw * alpha[j];
    float ga = gw * Tj[j];

    // recurrence contribution from T_{j+1}
    gT += gT_next * (1.0 - alpha[j]);
    ga += gT_next * (-Tj[j]);

    // OLD ALPHA CODE
    g_alpha_s[j] = unclamped_mask[j] ? ga : 0.0;

    gT_next = gT;
  }

  // -------- Pass 3: per-sample recompute contribs and accumulate parameter grads --------
  for (int32_t j = 0; j < N; ++j) {
    const int32_t idx = r * N + j;
    const float px = ray_xyz[idx * 3 + 0];
    const float py = ray_xyz[idx * 3 + 1];
    const float pz = ray_xyz[idx * 3 + 2];

    int ix = __float2int_rd((px - c_min_pos[0]) * c_inv_voxel);
    int iy = __float2int_rd((py - c_min_pos[1]) * c_inv_voxel);
    int iz = __float2int_rd((pz - c_min_pos[2]) * c_inv_voxel);

    if (ix < 0 || ix >= c_W || iy < 0 || iy >= c_H || iz < 0 || iz >= c_Z)
      continue;

    const int32_t vox = ix + c_W * (iy + c_H * iz);
    const int32_t count = voxel_counts[vox];
    if (count <= 0) continue;

    const int32_t start = voxel_offsets[vox];

    // w(j) = T(j)*alpha(j) (alpha already clamped)
    const float wj = Tj[j] * alpha[j];

    for (int32_t kk = 0; kk < count; ++kk) {
      const int32_t q = voxel_sq_ids[start + kk];

      // forward intermediates
      const float mx = means[q * 3 + 0];
      const float my = means[q * 3 + 1];
      const float mz = means[q * 3 + 2];

      const float dx = px - mx;
      const float dy = py - my;
      const float dz = pz - mz;

      const float* Rm = rots + q * 9;
      const float r0 = Rm[0], r1 = Rm[1], r2 = Rm[2];
      const float r3 = Rm[3], r4 = Rm[4], r5 = Rm[5];
      const float r6 = Rm[6], r7 = Rm[7], r8 = Rm[8];

      const float x = dx * r0 + dy * r1 + dz * r2;
      const float y = dx * r3 + dy * r4 + dz * r5;
      const float z = dx * r6 + dy * r7 + dz * r8;

      const float a = scale[q * 3 + 0];
      const float b = scale[q * 3 + 1];
      const float c = scale[q * 3 + 2];

      const float e1 = eps1[q];
      const float e2 = eps2[q];

      // contrib and superquad terms re-eval for gradients
      const float inv_a = 1.0 / a;
      const float inv_b = 1.0 / b;
      const float inv_c = 1.0 / c;

      // exponents (same clamp as forward)
      const float p = clampf(2.0 / e2, -1e9, 6.0);
      const float r = clampf(2.0 / e1, -1e9, 6.0);
      const float s = clampf(e2 / e1, -1e9, 6.0);

      const float Ax = clampf(fabsf(x * inv_a), 0.0, 50.0);
      const float Ay = clampf(fabsf(y * inv_b), 0.0, 50.0);
      const float Az = clampf(fabsf(z * inv_c), 0.0, 50.0);

      const float tx = __powf(Ax, p);
      const float ty = __powf(Ay, p);
      const float inner = tx + ty;
      const float xy = __powf(inner, s);
      const float zz = __powf(Az, r);

      const float f = xy + zz;
      const float contrib = __expf(-f);

      // g_sem_s[c] = g_sem[c] * w(j)
      // sem_pred grad: g += g_sem_s[c] * contrib
#pragma unroll
      for (int cc = 0; cc < C_SEM; ++cc) {
        const float gsem_s = g_sem[cc] * wj;
        atomicAdd(&grad_sem[q * C_SEM + cc], gsem_s * contrib);
      }

      // opacity grad: alpha_s += contrib * opacity
      atomicAdd(&grad_opacity[q], g_alpha_s[j] * contrib);

      // contrib grad:
      // g_contrib = g_alpha_s * opacity + sum_c (g_sem[c]*w) * sem_q[c]
      const float* sem_row = sem + q * C_SEM;
      float dot_gsem_semq = 0.0;
#pragma unroll
      for (int cc = 0; cc < C_SEM; ++cc) dot_gsem_semq += (g_sem[cc] * wj) * sem_row[cc];

      const float g_contrib = g_alpha_s[j] * opacity[q] + dot_gsem_semq;

      // contrib = exp(-f) => df = -g_contrib * contrib
      const float g_f = (-g_contrib) * contrib;

      // Now differentiate f = xy + zz

      // zz = Az^r
      // dzz/dAz = r*Az^(r-1), dzz/dr = zz*log(Az)
      const float Az_safe = fmax(Az, 1e-12);
      const float inner_safe = fmax(inner, 1e-12);
      const float Ax_safe = fmax(Ax, 1e-12);
      const float Ay_safe = fmax(Ay, 1e-12);

      float g_Az = g_f * (r * __powf(Az_safe, r - 1.0));
      float g_r  = g_f * (zz * __logf(Az_safe));

      // xy = inner^s
      float g_inner = g_f * (s * __powf(inner_safe, s - 1.0));
      float g_s     = g_f * (xy * __logf(inner_safe));

      // inner = tx + ty
      float g_tx = g_inner;
      float g_ty = g_inner;

      // tx = Ax^p, ty = Ay^p
      float g_Ax = g_tx * (p * __powf(Ax_safe, p - 1.0));
      float g_Ay = g_ty * (p * __powf(Ay_safe, p - 1.0));

      float g_p = g_tx * (tx * __logf(Ax_safe)) + g_ty * (ty * __logf(Ay_safe));

      // parameters p=2/e2, r=2/e1, s=e2/e1
      // dp/de2 = -2/e2^2
      // dr/de1 = -2/e1^2
      // ds/de2 = 1/e1, ds/de1 = -e2/e1^2
      float g_e1 = 0.0;
      float g_e2 = 0.0;

      g_e2 += g_p * (-2.0 / (e2 * e2));
      g_e1 += g_r * (-2.0 / (e1 * e1));

      g_e2 += g_s * (1.0 / e1);
      g_e1 += g_s * (-e2 / (e1 * e1));

      atomicAdd(&grad_eps1[q], g_e1);
      atomicAdd(&grad_eps2[q], g_e2);

      // Backprop through Ax=|x/a|, Ay=|y/b|, Az=|z/c|
      const float sx = sgnf(x * inv_a);
      const float sy = sgnf(y * inv_b);
      const float sz = sgnf(z * inv_c);

      // Ax = |x|/a
      float g_x = g_Ax * sx * (1.0 / a);
      float g_a = g_Ax * sx * (-(x) / (a * a));

      float g_y = g_Ay * sy * (1.0 / b);
      float g_b = g_Ay * sy * (-(y) / (b * b));

      float g_z = g_Az * sz * (1.0 / c);
      float g_c = g_Az * sz * (-(z) / (c * c));

      atomicAdd(&grad_scale[q * 3 + 0], g_a);
      atomicAdd(&grad_scale[q * 3 + 1], g_b);
      atomicAdd(&grad_scale[q * 3 + 2], g_c);

      // x,y,z = R * d
      // grads for R:
      atomicAdd(&grad_rots[q * 9 + 0], g_x * dx);
      atomicAdd(&grad_rots[q * 9 + 1], g_x * dy);
      atomicAdd(&grad_rots[q * 9 + 2], g_x * dz);

      atomicAdd(&grad_rots[q * 9 + 3], g_y * dx);
      atomicAdd(&grad_rots[q * 9 + 4], g_y * dy);
      atomicAdd(&grad_rots[q * 9 + 5], g_y * dz);

      atomicAdd(&grad_rots[q * 9 + 6], g_z * dx);
      atomicAdd(&grad_rots[q * 9 + 7], g_z * dy);
      atomicAdd(&grad_rots[q * 9 + 8], g_z * dz);

      // g_d = R^T * [g_x,g_y,g_z]
      const float gd0 = r0 * g_x + r3 * g_y + r6 * g_z;
      const float gd1 = r1 * g_x + r4 * g_y + r7 * g_z;
      const float gd2 = r2 * g_x + r5 * g_y + r8 * g_z;

      // d = p - mean => dmean = -g_d
      atomicAdd(&grad_means[q * 3 + 0], -gd0);
      atomicAdd(&grad_means[q * 3 + 1], -gd1);
      atomicAdd(&grad_means[q * 3 + 2], -gd2);
    }
  }
}

std::vector<torch::Tensor> sqocc_render_backward_cuda(
    torch::Tensor grad_out_sem,    // [R,C] float
    torch::Tensor grad_out_depth,  // [R]   float
    torch::Tensor ray_xyz,         // [R,N,3] float
    torch::Tensor ray_t,           // [R,N]   float
    torch::Tensor voxel_offsets,   // [V] int32
    torch::Tensor voxel_counts,    // [V] int32
    torch::Tensor voxel_sq_ids,    // [T] int32
    torch::Tensor means,           // [Q,3] float
    torch::Tensor rots,            // [Q,3,3] float (will view as [Q,9])
    torch::Tensor scale,           // [Q,3] float
    torch::Tensor eps1,            // [Q] float
    torch::Tensor eps2,            // [Q] float
    torch::Tensor opacity,         // [Q] float
    torch::Tensor sem_pred         // [Q,C] float
) {
  CHECK_INPUT(grad_out_sem);
  CHECK_INPUT(grad_out_depth);
  CHECK_INPUT(ray_xyz);
  CHECK_INPUT(ray_t);
  CHECK_INPUT(voxel_offsets);
  CHECK_INPUT(voxel_counts);
  CHECK_INPUT(voxel_sq_ids);
  CHECK_INPUT(means);
  CHECK_INPUT(rots);
  CHECK_INPUT(scale);
  CHECK_INPUT(eps1);
  CHECK_INPUT(eps2);
  CHECK_INPUT(opacity);
  CHECK_INPUT(sem_pred);

  TORCH_CHECK(ray_xyz.scalar_type() == at::kFloat, "ray_xyz must be float32");
  TORCH_CHECK(ray_t.scalar_type()   == at::kFloat, "ray_t must be float32");
  TORCH_CHECK(means.scalar_type()   == at::kFloat, "means must be float32");
  TORCH_CHECK(rots.scalar_type()    == at::kFloat, "rots must be float32");
  TORCH_CHECK(scale.scalar_type()   == at::kFloat, "scale must be float32");
  TORCH_CHECK(eps1.scalar_type()    == at::kFloat, "eps1 must be float32");
  TORCH_CHECK(eps2.scalar_type()    == at::kFloat, "eps2 must be float32");
  TORCH_CHECK(opacity.scalar_type() == at::kFloat, "opacity must be float32");
  TORCH_CHECK(sem_pred.scalar_type()== at::kFloat, "sem_pred must be float32");
  TORCH_CHECK(grad_out_sem.scalar_type() == at::kFloat, "grad_out_sem must be float32");
  TORCH_CHECK(grad_out_depth.scalar_type() == at::kFloat, "grad_out_depth must be float32");

  TORCH_CHECK(voxel_offsets.scalar_type() == at::kInt, "voxel_offsets must be int32");
  TORCH_CHECK(voxel_counts.scalar_type()  == at::kInt, "voxel_counts must be int32");
  TORCH_CHECK(voxel_sq_ids.scalar_type()  == at::kInt, "voxel_sq_ids must be int32");

  const int32_t R = (int32_t)ray_xyz.size(0);
  const int32_t N = (int32_t)ray_xyz.size(1);
  const int32_t Q = (int32_t)means.size(0);

  // flatten rots to [Q,9]
  auto rots_f = rots.contiguous().view({Q, 9});

  auto grad_means   = torch::zeros_like(means);
  auto grad_rots    = torch::zeros_like(rots_f);
  auto grad_scale   = torch::zeros_like(scale);
  auto grad_eps1    = torch::zeros_like(eps1);
  auto grad_eps2    = torch::zeros_like(eps2);
  auto grad_opacity = torch::zeros_like(opacity);
  auto grad_sem     = torch::zeros_like(sem_pred);

  const int threads = 256;
  const int blocks = (R + threads - 1) / threads;

  sqocc_render_backward_kernel<<<blocks, threads>>>(
      (const float *)ray_xyz.data_ptr<float>(),
      (const float *)ray_t.data_ptr<float>(),
      R, N,
      (const int32_t *)voxel_offsets.data_ptr<int32_t>(),
      (const int32_t *)voxel_counts.data_ptr<int32_t>(),
      (const int32_t *)voxel_sq_ids.data_ptr<int32_t>(),
      (const float *)means.data_ptr<float>(),
      (const float *)rots_f.data_ptr<float>(),
      (const float *)scale.data_ptr<float>(),
      (const float *)eps1.data_ptr<float>(),
      (const float *)eps2.data_ptr<float>(),
      (const float *)opacity.data_ptr<float>(),
      (const float *)sem_pred.data_ptr<float>(),
      (const float *)grad_out_sem.data_ptr<float>(),
      (const float *)grad_out_depth.data_ptr<float>(),
      (float *)grad_means.data_ptr<float>(),
      (float *)grad_rots.data_ptr<float>(),
      (float *)grad_scale.data_ptr<float>(),
      (float *)grad_eps1.data_ptr<float>(),
      (float *)grad_eps2.data_ptr<float>(),
      (float *)grad_opacity.data_ptr<float>(),
      (float *)grad_sem.data_ptr<float>()
    );

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "backward kernel launch failed: ", cudaGetErrorString(err));

  // reshape rots grad back to [Q,3,3]
  auto grad_rots_3x3 = grad_rots.view({Q, 3, 3});

  return {grad_means, grad_rots_3x3, grad_scale, grad_eps1, grad_eps2, grad_opacity, grad_sem, 
  };
}
