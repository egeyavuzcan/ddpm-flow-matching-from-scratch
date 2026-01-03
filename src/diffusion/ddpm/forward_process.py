"""
DDPM Forward Process.

Implements the forward diffusion process: q(x_t | x_0)
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

from .noise_schedule import NoiseSchedule


class DDPMForwardProcess(nn.Module):
    """
    DDPM Forward Process: Add noise to data.
    
    Forward process: q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
    
    Reparameterization: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    """
    
    def __init__(self, schedule: NoiseSchedule):
        """
        Args:
            schedule: Precomputed noise schedule
        """
        super().__init__()
        self.schedule = schedule
        self.num_timesteps = schedule.num_timesteps
    
    def add_noise(
        self,
        x_0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Add noise to clean data at timestep t.
        
        x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        
        Args:
            x_0: (B, C, H, W) clean data in range [-1, 1]
            t: (B,) timestep indices (0 to T-1)
            noise: (B, C, H, W) optional pre-sampled noise
        
        Returns:
            x_t: (B, C, H, W) noisy data
            noise: (B, C, H, W) the noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get coefficients at timestep t
        sqrt_alpha_bar = self.schedule.get_value(
            self.schedule.sqrt_alpha_bar, t, x_0.shape
        )
        sqrt_one_minus_alpha_bar = self.schedule.get_value(
            self.schedule.sqrt_one_minus_alpha_bar, t, x_0.shape
        )
        
        # Forward process
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise
    
    def get_target(
        self,
        x_0: Tensor,
        noise: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Get training target for the model.
        
        For DDPM, the target is simply the noise ε.
        
        Args:
            x_0: Clean data (not used, but included for interface consistency)
            noise: The noise that was added
            t: Timestep (not used for simple noise prediction)
        
        Returns:
            Target for the model (the noise)
        """
        return noise
    
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
            (B,) random timesteps in [0, T-1]
        """
        return torch.randint(
            0, self.num_timesteps, (batch_size,), device=device
        )
