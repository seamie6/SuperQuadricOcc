import torch, torch.nn as nn

from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    PerspectiveCameras,
    NDCMultinomialRaysampler,
)

from pytorch3d.renderer.implicit.utils import ray_bundle_to_ray_points
# from .superquadricocc_render import superquadricocc_render
from mmdet3d.models.detectors import superquadricocc_render

class SuperQuadricOccRender(nn.Module):
    def __init__(self, grid_cfg, N_samples, image_size):
        super().__init__()

        # configs
        self.N_samples = N_samples

        # render limit 
        self.near, self.far = 0.1, 40.0

        t_vals = torch.linspace(
            self.near,
            self.far,
            steps=self.N_samples,
            dtype=torch.float32,
        )

        self.t_vals = t_vals
        self.image_size = torch.tensor([[image_size[0], image_size[1]]])

        # voxel stuffs
        self.C = 16
        self.W, self.H, self.Z = 200,200,16
        self.grid_cfg = grid_cfg

        # ray sampler
        self.raysampler = NDCMultinomialRaysampler(
            image_width=image_size[1],
            image_height=image_size[0],
            n_pts_per_ray=self.N_samples,
            min_depth=self.near,
            max_depth=self.far,
        )

    def forward(self, means, eps1, eps2, scale, rots, opacity, sem_pred,
                gs_intrins, gs_extrins, voxel_offsets, voxel_counts, voxel_sq_ids, device):

        t_vals = self.t_vals.to(device)
        image_size = self.image_size.to(device)

        # TODO: move intrinsic setup to CUDA for speedup
        R_cv = gs_extrins[:, :3, :3]
        t_cv = gs_extrins[:, :3, 3]

        fx = gs_intrins[:,0,0]
        fy = gs_intrins[:,1,1]
        cx = gs_intrins[:,0,2]
        cy = gs_intrins[:,1,2]

        tt = t_cv
        RR = R_cv.permute(0, 2, 1)
        f = torch.stack([fx, fy], dim=-1)
        p = torch.stack([cx, cy], dim=-1)

        cameras = PerspectiveCameras(R=RR, T=tt, focal_length=-f, principal_point=p, image_size=image_size, in_ndc=False,
                                     device=device)

        ray_bundle = self.raysampler(
            cameras=cameras
        )

        ray_points = ray_bundle_to_ray_points(ray_bundle)  # [B,H,W,N,3]

        ray_pts = ray_points.reshape(-1, self.N_samples, 3).contiguous()
        means = means.contiguous()
        
        # CUDA pass
        inputs = (
            ray_pts, t_vals,
            voxel_offsets, voxel_counts, voxel_sq_ids,
            means, rots, scale, eps1, eps2, opacity, sem_pred
        )
        out_sem, out_depth = SuperQuadricOccRenderFN.apply(*inputs)

        sem   = out_sem.reshape(ray_points.shape[:-2] + (self.C,))         # [6,H,W,16]
        depth = out_depth.reshape(ray_points.shape[:-2])              # [6,H,W]

        return sem.squeeze(0), depth.squeeze(0)
    
class SuperQuadricOccRenderFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                ray_xyz, ray_t,
                voxel_offsets, voxel_counts, voxel_sq_ids,
                means, rots, scale, eps1, eps2, opacity, sem_pred
                ):

        out_sem, out_depth = superquadricocc_render.raymarch_forward(
            ray_xyz, ray_t,
            voxel_offsets, voxel_counts, voxel_sq_ids,
            means, rots, scale, eps1, eps2, opacity, sem_pred
        )

        ctx.save_for_backward(
            ray_xyz, ray_t,
            voxel_offsets, voxel_counts, voxel_sq_ids,
            means, rots, scale, eps1, eps2, opacity, sem_pred
        )

        return out_sem, out_depth

    @staticmethod
    def backward(ctx, grad_out_sem, grad_out_depth):
        (ray_xyz, ray_t,
         voxel_offsets, voxel_counts, voxel_sq_ids,
         means, rots, scale, eps1, eps2, opacity, sem_pred) = ctx.saved_tensors
        
        grad_out_sem = grad_out_sem.contiguous()
        grad_out_depth = grad_out_depth.contiguous()

        grads = superquadricocc_render.raymarch_backward(
            grad_out_sem, grad_out_depth,
            ray_xyz, ray_t,
            voxel_offsets, voxel_counts, voxel_sq_ids,
            means, rots, scale, eps1, eps2, opacity, sem_pred
        )
        
        (
            grad_means,
            grad_rots,
            grad_scale,
            grad_eps1,
            grad_eps2,
            grad_opacity,
            grad_sem_pred,
        ) = grads

        return (
            None,  # ray_xyz
            None,  # ray_t
            None,  # voxel_offsets
            None,  # voxel_counts
            None,  # voxel_sq_ids
            grad_means,
            grad_rots,
            grad_scale,
            grad_eps1,
            grad_eps2,
            grad_opacity,
            grad_sem_pred
        )