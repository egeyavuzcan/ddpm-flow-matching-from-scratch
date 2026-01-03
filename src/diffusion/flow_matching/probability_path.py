"""
Probability path for Flow Matching.

Implements linear interpolation between data and noise:
    x_t = (1 - t) · x_0 + t · x_1
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional


class FlowMatchingPath(nn.Module):
    """
    Linear probability path for Flow Matching.
    
    Path: x_t = (1 - t) · x_0 + t · x_1
    
    where:
    - x_0 = data sample
    - x_1 = noise sample ~ N(0, I)
    - t ∈ [0, 1]
    """
    
    def __init__(self, sigma_min: float = 0.0):
        """
        Args:
            sigma_min: Minimum noise level (can add small noise even at t=0)
        """
        super().__init__()
        self.sigma_min = sigma_min
    
    def interpolate(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Interpolate between data and noise.
        
        x_t = (1 - t) · x_0 + t · x_1
        
        Args:
            x_0: (B, C, H, W) data samples
            x_1: (B, C, H, W) noise samples
            t: (B,) or (B, 1, 1, 1) timesteps in [0, 1]
        
        Returns:
            x_t: (B, C, H, W) interpolated samples
        """
        # Reshape t for broadcasting
        if t.dim() == 1:
            t = t[:, None, None, None]
        
        return (1 - t) * x_0 + t * x_1
    
    def get_noisy_sample(
        self,
        x_0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Create noisy sample at timestep t.
        
        This is the main interface used during training, analogous to
        DDPMForwardProcess.add_noise()
        
        Args:
            x_0: (B, C, H, W) data samples
            t: (B,) timesteps in [0, 1]
            noise: (B, C, H, W) optional pre-sampled noise (x_1)
        
        Returns:
            x_t: Interpolated sample
            noise: The noise (x_1) that was used
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        x_t = self.interpolate(x_0, noise, t)
        
        return x_t, noise
    
    def get_target(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Get the velocity target for training.
        
        For linear path, velocity is constant: u = x_1 - x_0
        
        Args:
            x_0: Data samples
            x_1: Noise samples
            t: Timestep (not used for linear path, but included for interface)
        
        Returns:
            Velocity target (x_1 - x_0)
        """
        return x_1 - x_0
    
    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """
        Sample random timesteps for training.
        
        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on
        
        Returns:
            (B,) random timesteps in [0, 1]
        """
        return torch.rand(batch_size, device=device)
