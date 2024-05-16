"""Depth utilities."""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

def depth_to_disparity(depth: Float[Tensor, "BS 1 H W"], min_percentile: float = 0.05, max_percentile: float = 0.95, eps: float = 1e-6):
    """Convert depth to disparity. We normalize according to Eq. 10 in this paper https://arxiv.org/pdf/2401.05583.pdf."""
    assert depth.dim() == 4, "Depth must be of shape (BS, 1, H, W)"
    BS = depth.shape[0]
    disparity = 1 / (depth + eps)
    if min_percentile == 0 and max_percentile == 1:
        return disparity
    disparity_min = torch.quantile(disparity.view(BS, -1).float(), min_percentile, dim=1, interpolation='nearest')
    disparity_max = torch.quantile(disparity.view(BS, -1).float(), max_percentile, dim=1, interpolation='nearest')
    disparity_min = disparity_min.view(BS, 1, 1, 1)
    disparity_max = disparity_max.view(BS, 1, 1, 1)
    disparity = (disparity - disparity_min) / (disparity_max - disparity_min + eps)
    return disparity

def create_discontinuity_mask(depths, threshold=0.05, dilation_kernel_size=3):
    """
    Create a mask for pixels with large depth discontinuities and apply dilation, with a default threshold.
    
    Args:
    depths (torch.Tensor): Tensor of shape [N, H, W] containing N depth maps.
    threshold (float, optional): Threshold for gradient magnitude to identify discontinuities. Default is 0.1.
    dilation_kernel_size (int): Size of the kernel used for dilation (must be odd).
    
    Returns:
    torch.Tensor: Binary mask of shape [N, H, W] where 1 indicates a large discontinuity, dilated.
    """
    # Sobel operator kernels for gradient computation in x and y directions
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    # Ensure the kernels are on the same device as the input depths
    sobel_x = sobel_x.to(depths.device)
    sobel_y = sobel_y.to(depths.device)
    
    depths = depths.unsqueeze(1)  # Add a channel dimension for processing
    
    # Compute gradients in x and y directions
    grad_x = F.conv2d(depths, sobel_x, padding=1)
    grad_y = F.conv2d(depths, sobel_y, padding=1)
    
    # Compute gradient magnitude
    grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    
    # Create binary mask for large discontinuities
    mask = grad_magnitude > threshold
    
    # Dilate the mask using max pooling as a proxy for dilation
    if dilation_kernel_size > 1:
        # Apply max pooling to dilate. The output size is matched to the input by adjusting padding.
        dilated_mask = F.max_pool2d(mask.float(), kernel_size=dilation_kernel_size, stride=1, padding=dilation_kernel_size//2)
        mask = dilated_mask > 0  # Convert back to a binary mask
    
    # Ensure the output dimensions match the input [N, H, W]
    mask = mask.squeeze(1)
    
    return mask
