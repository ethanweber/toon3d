"""Loss functions."""

import torch
from jaxtyping import Float
from torch import Tensor

def depth_ranking_loss(
    rendered_depth: Float[Tensor, "*batch H W"],
    gt_depth: Float[Tensor, "*batch H W"],
    mask: Float[Tensor, "*batch H W"],
    patch_size: int = 128,
    num_patches: int = 8,
    epsilon: float = 1e-6,
) -> Float[Tensor, "*batch"]:
    """
    Depth ranking loss as described in the SparseGS paper.

    Args:
        rendered_depth: rendered depth image
        gt_depth: ground truth depth image
        mask: mask for the depth images. 1 where valid, 0 where invalid
        patch_size: patch size
        num_patches: number of patches to sample
        epsilon: small value to avoid division by zero
    """

    b, h, w = rendered_depth.shape

    # construct patch indices
    sh = torch.randint(0, h - patch_size, (b, num_patches, patch_size, patch_size))
    sw = torch.randint(0, w - patch_size, (b, num_patches, patch_size, patch_size))
    idx_batch = torch.arange(b)[:, None, None, None].repeat(1, num_patches, patch_size, patch_size)
    idx_rows = torch.arange(patch_size)[None, None, None, :].repeat(b, 1, 1, 1) + sh
    idx_cols = torch.arange(patch_size)[None, None, None, :].repeat(b, 1, 1, 1) + sw

    # index into and mask out patches
    mask_patches = mask[idx_batch, idx_rows, idx_cols]
    rendered_depth_patches = rendered_depth[idx_batch, idx_rows, idx_cols] * mask_patches
    gt_depth_patches = gt_depth[idx_batch, idx_rows, idx_cols] * mask_patches

    # calculate correlation
    e_xy = torch.mean(rendered_depth_patches * gt_depth_patches, dim=[-1, -2])
    e_x = torch.mean(rendered_depth_patches, dim=[-1, -2])
    e_y = torch.mean(gt_depth_patches, dim=[-1, -2])
    e_x2 = torch.mean(torch.square(rendered_depth_patches), dim=[-1, -2])
    ex_2 = e_x**2
    e_y2 = torch.mean(torch.square(gt_depth_patches), dim=[-1, -2])
    ey_2 = e_y**2
    corr = (e_xy - e_x * e_y) / (torch.sqrt((e_y2 - ey_2) * (e_x2 - ex_2)) + epsilon)
    corr = torch.clamp(corr, min=-1, max=1)

    # calculate loss
    loss = torch.mean(1 - corr, dim=-1)
    return loss