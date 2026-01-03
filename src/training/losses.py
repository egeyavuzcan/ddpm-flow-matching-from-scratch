"""
Loss functions for diffusion models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SimpleMSELoss(nn.Module):
    """
    Simple MSE loss for diffusion models.
    
    Used for:
    - DDPM: MSE(ε, ε_θ) - comparing predicted noise to actual noise
    - Flow Matching: MSE(v, v_θ) - comparing predicted velocity to target
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Args:
            reduction: "mean", "sum", or "none"
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute MSE loss.
        
        Args:
            pred: (B, C, H, W) predicted noise/velocity
            target: (B, C, H, W) target noise/velocity
        
        Returns:
            Scalar loss (if reduction="mean" or "sum")
        """
        return F.mse_loss(pred, target, reduction=self.reduction)


def ddpm_loss(
    model: nn.Module,
    x_0: Tensor,
    t: Tensor,
    noise: Tensor,
    x_t: Tensor,
    class_label: Tensor = None,
) -> Tensor:
    """
    Compute DDPM training loss.
    
    L = ||ε - ε_θ(x_t, t)||²
    
    Args:
        model: UNet model
        x_0: Clean data (not used, but for signature consistency)
        t: Timesteps
        noise: The noise that was added
        x_t: Noisy data
        class_label: Optional class labels
    
    Returns:
        Scalar MSE loss
    """
    noise_pred = model(x_t, t, class_label)
    return F.mse_loss(noise_pred, noise)


def flow_matching_loss(
    model: nn.Module,
    x_0: Tensor,
    t: Tensor,
    x_1: Tensor,
    x_t: Tensor,
    class_label: Tensor = None,
) -> Tensor:
    """
    Compute Flow Matching training loss.
    
    L = ||v_θ(x_t, t) - (x_1 - x_0)||²
    
    Args:
        model: UNet model
        x_0: Data samples
        t: Timesteps
        x_1: Noise samples
        x_t: Interpolated samples
        class_label: Optional class labels
    
    Returns:
        Scalar MSE loss
    """
    v_pred = model(x_t, t, class_label)
    target = x_1 - x_0
    return F.mse_loss(v_pred, target)
