"""
Noise schedule for DDPM.

Implements beta, alpha, and alpha_bar calculations for different schedule types.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal
import math


def linear_beta_schedule(
    num_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> Tensor:
    """
    Linear beta schedule as in original DDPM paper.
    
    β_t = β_start + (β_end - β_start) * t / T
    
    Args:
        num_timesteps: Number of diffusion steps T
        beta_start: Starting beta value
        beta_end: Ending beta value
    
    Returns:
        (T,) tensor of beta values
    """
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(
    num_timesteps: int = 1000,
    s: float = 0.008,
) -> Tensor:
    """
    Cosine beta schedule as in Improved DDPM paper.
    
    Provides smoother noise schedule, especially at low timesteps.
    
    Args:
        num_timesteps: Number of diffusion steps T
        s: Small offset to prevent β_0 = 0
    
    Returns:
        (T,) tensor of beta values
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    
    # Compute alpha_bar using cosine schedule
    alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]  # Normalize so alpha_bar_0 = 1
    
    # Compute betas from alpha_bar
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    
    # Clip to prevent numerical issues
    return torch.clamp(betas, 0.0001, 0.9999)


class NoiseSchedule(nn.Module):
    """
    Precomputed noise schedule for DDPM.
    
    Stores all necessary coefficients for forward and reverse processes:
    - beta_t: Variance schedule
    - alpha_t: 1 - beta_t
    - alpha_bar_t: Cumulative product of alpha
    - sqrt_alpha_bar_t: For forward process
    - sqrt_one_minus_alpha_bar_t: For forward process
    - And more for reverse process...
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: Literal["linear", "cosine"] = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        """
        Args:
            num_timesteps: Number of diffusion steps T
            schedule_type: Type of schedule ("linear" or "cosine")
            beta_start: Starting beta (for linear schedule)
            beta_end: Ending beta (for linear schedule)
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # Compute betas
        if schedule_type == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Compute derived quantities
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])
        
        # Register as buffers (not parameters, but should be on same device)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        
        # Forward process coefficients
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        
        # Reverse process coefficients
        self.register_buffer("sqrt_recip_alpha", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recip_alpha_bar", torch.sqrt(1.0 / alpha_bar))
        self.register_buffer("sqrt_recip_alpha_bar_minus_one", torch.sqrt(1.0 / alpha_bar - 1))
        
        # Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_variance[0] = betas[0]  # First step variance
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        # Posterior mean coefficients
        # μ̃_t = (√ᾱ_{t-1} β_t)/(1-ᾱ_t) x_0 + (√α_t (1-ᾱ_{t-1}))/(1-ᾱ_t) x_t
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_bar_prev) * torch.sqrt(alphas) / (1.0 - alpha_bar)
        )
    
    def get_value(self, values: Tensor, t: Tensor, shape: tuple) -> Tensor:
        """
        Extract values at timestep t and reshape for broadcasting.
        
        Args:
            values: (T,) tensor of precomputed values
            t: (B,) tensor of timestep indices
            shape: Shape to broadcast to (e.g., (B, C, H, W))
        
        Returns:
            Values indexed by t, reshaped for broadcasting
        """
        batch_size = t.shape[0]
        out = values.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))
