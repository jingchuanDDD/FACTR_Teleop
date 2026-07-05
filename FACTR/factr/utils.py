# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------


import math
import numpy as np
import torch
import torch.nn.functional as F

def gaussian_2d_kernel(kernel_size: int, sigma: float, device=None, dtype=None) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for convolution.
    
    Args:
        kernel_size: integer, the height/width of the kernel (assumed square).
        sigma: standard deviation for the Gaussian.
        device, dtype: optional, to place the kernel on a specific device / dtype.
    Returns:
        kernel: Tensor of shape (kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords -= (kernel_size - 1) / 2.0  # shift to center
    x, y = torch.meshgrid(coords, coords, indexing='xy')
    kernel_2d = torch.exp(-0.5 * (x**2 + y**2) / sigma**2)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

def gaussian_2d_smoothing(
    img: torch.Tensor,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Apply 2D Gaussian smoothing (blur) to a batch of images.
    
    Args:
        img: Tensor of shape (..., C, H, W).
        scale: Controls the standard deviation (sigma) of the Gaussian kernel. Larger scale corresponds to more smoothing.

    Returns:
        blurred: Tensor of the same shape as img.
    """
    if scale <= 0:
        return img

    sigma = scale
    kernel_size = max(3, 2 * math.ceil(3 * sigma) + 1)

    kernel_2d = gaussian_2d_kernel(kernel_size, sigma, device=img.device, dtype=img.dtype)
    kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)

    C = img.shape[-3]
    kernel_2d = kernel_2d.repeat(C, 1, 1, 1)  # shape: (C, 1, kH, kW)

    padding = kernel_size // 2
    
    original_shape = img.shape
    batch_shape = original_shape[:-3]
    spatial_shape = original_shape[-2:]
    batch_size = int(torch.prod(torch.tensor(batch_shape)))
    img_reshaped = img.view(batch_size, C, *spatial_shape)
    
    blurred = F.conv2d(img_reshaped, kernel_2d, groups=C, padding=padding)
    return blurred.view(*original_shape)


def gaussian_1d_smoothing(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Apply Gaussian 1D smoothing across the last dimension (feature_dim).
    
    Args:
        x: Tensor of shape (..., feature_dim).
        scale: Controls the standard deviation (sigma) of the Gaussian kernel. Larger scale corresponds to more smoothing.
    
    Returns:
        smoothed_x: Tensor of the same shape as x, but blurred along the last dimension.
    """
    # Handle edge case: if scale is very small, just return x
    if scale <= 0:
        return x

    # kernel_size = 2 * int(3*sigma) + 1 as a rule-of-thumb.
    sigma = scale
    kernel_size = max(3, 2 * int(3 * sigma) + 1)  # ensure at least 3

    half_size = (kernel_size - 1) // 2
    arange = torch.arange(-half_size, half_size + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-0.5 * (arange / sigma)**2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_1d = kernel_1d.view(1, 1, -1)
    
    padding = half_size

    original_shape = x.shape
    feature_dim = x.shape[-1]
    batch_size = int(torch.prod(torch.tensor(x.shape[:-1])))
    x_reshaped = x.view(batch_size, 1, feature_dim)
    
    smoothed = F.conv1d(x_reshaped, kernel_1d, padding=padding)
    smoothed_x = smoothed.view(*original_shape)
    
    return smoothed_x

def downsample_1d(x, scale=2):
    scale = int(np.round(scale))
    if scale <= 1:
        return x
    original_shape = x.shape
    x_down = F.avg_pool1d(x.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)  # (B, K/2)
    x_up = F.interpolate(x_down.unsqueeze(1), size=original_shape[-1], mode='nearest').squeeze(1)  # or 'linear'
    return x_up

def downsample_2d(img, scale=2):
    scale = int(np.round(scale))
    if scale <= 1:
        return img
    original_shape = img.shape
    x_down = F.avg_pool2d(img, kernel_size=scale, stride=scale)  # (B, C, H/2, W/2)
    x_up = F.interpolate(x_down, size=original_shape[-2:], mode='nearest')  # or 'bilinear'
    return x_up

def get_scale(scheduler, start, end, cur_step, max_step, ratio=2/3):
    assert start >= end, "Start scale must be larger than end scale"
    assert cur_step <= max_step, "Current step must be less than or equal to max step"
    assert cur_step >= 0 and max_step > 0, "Steps must be non-negative and max_step must be positive"
    
    if scheduler == 'no':
        return 0
    
    t = cur_step / max_step
    if t <= ratio or scheduler == 'const':
        return start
    
    t_rescaled = (t - ratio) / (1 - ratio)
    if scheduler == 'linear':
        scale = start + t_rescaled * (end - start)
    elif scheduler == 'cos':
        scale = end + 0.5 * (start - end) * (1 + np.cos(t_rescaled * np.pi))
    elif scheduler == 'exp':
        scale = start * np.exp(-5 * t_rescaled)
    elif scheduler == 'step':
        steps = 10
        step_index = int(t_rescaled * steps)
        scale = start + step_index * (end - start) / steps
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler}")
    
    return scale