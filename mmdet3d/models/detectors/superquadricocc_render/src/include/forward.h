#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> sqocc_render_forward_cuda(
    torch::Tensor ray_xyz,
    torch::Tensor ray_t,

    torch::Tensor voxel_offsets,
    torch::Tensor voxel_counts,
    torch::Tensor voxel_sq_ids,

    torch::Tensor means,
    torch::Tensor rots,
    torch::Tensor scale,
    torch::Tensor eps1,
    torch::Tensor eps2,
    torch::Tensor opacity,
    torch::Tensor sem_pred
);
