"""
DDPM Sampler.

Generates samples using the reverse diffusion process.
"""
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from typing import Optional

from diffusion.ddpm.noise_schedule import NoiseSchedule


class DDPMSampler:
    """
    DDPM Sampler for generating images.
    
    Uses the reverse process:
        p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)
    
    where:
        μ_θ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε_θ(x_t, t))
    """
    
    def __init__(
        self,
        model: nn.Module,
        schedule: NoiseSchedule,
        device: torch.device,
    ):
        """
        Args:
            model: Trained UNet model
            schedule: Noise schedule (same as used for training)
            device: Device to run on
        """
        self.model = model.to(device)
        self.schedule = schedule.to(device)
        self.device = device
        self.num_timesteps = schedule.num_timesteps
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_size: int = 32,
        channels: int = 3,
        class_label: Optional[Tensor] = None,
        show_progress: bool = True,
        num_inference_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate samples using the reverse process.
        
        Args:
            batch_size: Number of samples to generate
            image_size: Image resolution
            channels: Number of channels (3 for RGB)
            class_label: (B,) class labels for conditional generation
            show_progress: Show progress bar
            num_inference_steps: Number of steps to use (default: use all timesteps)
                                If less than num_timesteps, will skip steps uniformly
        
        Returns:
            (B, C, H, W) generated images in [-1, 1]
        """
        self.model.eval()
        
        # Start from pure noise
        shape = (batch_size, channels, image_size, image_size)
        x = torch.randn(shape, device=self.device)
        
        # Move class labels to device if provided
        if class_label is not None:
            class_label = class_label.to(self.device)
        
        # Determine timestep sequence
        if num_inference_steps is None or num_inference_steps >= self.num_timesteps:
            # Use all timesteps
            timesteps = list(reversed(range(self.num_timesteps)))
        else:
            # Skip timesteps uniformly
            stride = self.num_timesteps // num_inference_steps
            timesteps = list(range(self.num_timesteps - 1, 0, -stride))
        
        if show_progress:
            timesteps = tqdm(timesteps, desc=f"DDPM ({len(timesteps)} steps)")
        
        for t in timesteps:
            x = self._reverse_step(x, t, class_label)
        
        return x
    
    def _reverse_step(
        self,
        x_t: Tensor,
        t: int,
        class_label: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Single reverse step: x_t → x_{t-1}
        
        Args:
            x_t: Current noisy sample
            t: Current timestep
            class_label: Optional class labels
        
        Returns:
            x_{t-1}: Denoised sample
        """
        batch_size = x_t.shape[0]
        
        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        
        # Predict noise
        noise_pred = self.model(x_t, t_tensor, class_label)
        
        # Get coefficients
        alpha = self.schedule.alphas[t]
        alpha_bar = self.schedule.alpha_bar[t]
        beta = self.schedule.betas[t]
        
        # Compute mean
        # μ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε_θ)
        sqrt_recip_alpha = 1.0 / torch.sqrt(alpha)
        noise_coef = beta / torch.sqrt(1.0 - alpha_bar)
        mean = sqrt_recip_alpha * (x_t - noise_coef * noise_pred)
        
        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean
        
        return x_prev
    
    @torch.no_grad()
    def sample_with_trajectory(
        self,
        batch_size: int = 1,
        image_size: int = 32,
        channels: int = 3,
        class_label: Optional[Tensor] = None,
        save_every: int = 100,
    ) -> tuple:
        """
        Generate samples and save intermediate steps.
        
        Useful for visualization.
        
        Args:
            batch_size: Number of samples
            image_size: Image resolution
            channels: Number of channels
            class_label: Optional class labels
            save_every: Save every N steps
        
        Returns:
            (final_samples, trajectory) where trajectory is list of intermediate samples
        """
        self.model.eval()
        
        shape = (batch_size, channels, image_size, image_size)
        x = torch.randn(shape, device=self.device)
        
        if class_label is not None:
            class_label = class_label.to(self.device)
        
        trajectory = [x.cpu().clone()]
        
        for t in tqdm(reversed(range(self.num_timesteps)), desc="Sampling"):
            x = self._reverse_step(x, t, class_label)
            
            if t % save_every == 0:
                trajectory.append(x.cpu().clone())
        
        trajectory.append(x.cpu().clone())
        
        return x, trajectory
