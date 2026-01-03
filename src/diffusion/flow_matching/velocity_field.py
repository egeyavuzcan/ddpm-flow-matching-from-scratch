"""
Velocity field for Flow Matching.

For linear probability path:
    x_t = (1 - t) · x_0 + t · x_1

The conditional velocity field is:
    u_t(x_t | x_0, x_1) = x_1 - x_0

This is constant (independent of t) for linear interpolation!
"""
import torch
from torch import Tensor


def get_velocity_target(x_0: Tensor, x_1: Tensor) -> Tensor:
    """
    Compute the target velocity for Flow Matching.
    
    For linear interpolation path:
        x_t = (1 - t) · x_0 + t · x_1
    
    The velocity is:
        u = dx_t/dt = x_1 - x_0
    
    Args:
        x_0: (B, C, H, W) data samples
        x_1: (B, C, H, W) noise samples
    
    Returns:
        (B, C, H, W) velocity target
    """
    return x_1 - x_0


def compute_flow_matching_loss(
    v_pred: Tensor,
    x_0: Tensor,
    x_1: Tensor,
) -> Tensor:
    """
    Compute Flow Matching loss.
    
    L = ||v_θ(x_t, t) - (x_1 - x_0)||²
    
    Args:
        v_pred: (B, C, H, W) predicted velocity from model
        x_0: (B, C, H, W) data samples
        x_1: (B, C, H, W) noise samples
    
    Returns:
        Scalar loss value (mean squared error)
    """
    target = get_velocity_target(x_0, x_1)
    return torch.mean((v_pred - target) ** 2)
