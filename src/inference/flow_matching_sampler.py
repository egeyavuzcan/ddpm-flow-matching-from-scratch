"""
Flow Matching Sampler.

Generates samples by solving the ODE from noise to data.
"""
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from typing import Optional, Literal


class FlowMatchingSampler:
    """
    Flow Matching Sampler for generating images.
    
    Solves the ODE:
        dx/dt = v_θ(x, t)
    
    from t=1 (noise) to t=0 (data) using Euler method.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        """
        Args:
            model: Trained UNet model
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_size: int = 32,
        channels: int = 3,
        num_steps: int = 50,
        class_label: Optional[Tensor] = None,
        show_progress: bool = True,
        solver: Literal["euler", "heun"] = "euler",
    ) -> Tensor:
        """
        Generate samples using ODE solving.
        
        Args:
            batch_size: Number of samples to generate
            image_size: Image resolution
            channels: Number of channels (3 for RGB)
            num_steps: Number of ODE solver steps (20-50 typical)
            class_label: (B,) class labels for conditional generation
            show_progress: Show progress bar
            solver: ODE solver ("euler" or "heun")
        
        Returns:
            (B, C, H, W) generated images in [-1, 1]
        """
        self.model.eval()
        
        # Start from pure noise (at t=1)
        shape = (batch_size, channels, image_size, image_size)
        x = torch.randn(shape, device=self.device)
        
        # Move class labels to device if provided
        if class_label is not None:
            class_label = class_label.to(self.device)
        
        # Time steps: 1.0 → 0.0
        dt = 1.0 / num_steps
        timesteps = torch.linspace(1.0, dt, num_steps, device=self.device)
        
        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for t in timesteps:
            if solver == "euler":
                x = self._euler_step(x, t, dt, class_label)
            elif solver == "heun":
                x = self._heun_step(x, t, dt, class_label)
            else:
                raise ValueError(f"Unknown solver: {solver}")
        
        return x
    
    def _euler_step(
        self,
        x: Tensor,
        t: float,
        dt: float,
        class_label: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Single Euler step: x_{t-dt} = x_t - v_θ(x_t, t) * dt
        
        Args:
            x: Current sample
            t: Current time
            dt: Time step size
            class_label: Optional class labels
        
        Returns:
            Next sample
        """
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device)
        
        # Predict velocity
        v = self.model(x, t_tensor, class_label)
        
        # Euler step (going from t → t-dt, so we subtract)
        x = x - v * dt
        
        return x
    
    def _heun_step(
        self,
        x: Tensor,
        t: float,
        dt: float,
        class_label: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Heun's method (2nd order): More accurate than Euler.
        
        k1 = v(x, t)
        k2 = v(x - k1*dt, t-dt)
        x_next = x - (k1 + k2) * dt / 2
        """
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device)
        t_next = torch.full((batch_size,), t - dt, device=self.device)
        
        # First evaluation
        k1 = self.model(x, t_tensor, class_label)
        
        # Euler prediction
        x_pred = x - k1 * dt
        
        # Second evaluation
        k2 = self.model(x_pred, t_next, class_label)
        
        # Heun update
        x = x - (k1 + k2) * dt / 2
        
        return x
    
    @torch.no_grad()
    def sample_with_trajectory(
        self,
        batch_size: int = 1,
        image_size: int = 32,
        channels: int = 3,
        num_steps: int = 50,
        class_label: Optional[Tensor] = None,
    ) -> tuple:
        """
        Generate samples and save intermediate steps.
        
        Useful for visualization.
        
        Args:
            batch_size: Number of samples
            image_size: Image resolution
            channels: Number of channels
            num_steps: Number of ODE steps
            class_label: Optional class labels
        
        Returns:
            (final_samples, trajectory) where trajectory is list of intermediate samples
        """
        self.model.eval()
        
        shape = (batch_size, channels, image_size, image_size)
        x = torch.randn(shape, device=self.device)
        
        if class_label is not None:
            class_label = class_label.to(self.device)
        
        trajectory = [x.cpu().clone()]
        
        dt = 1.0 / num_steps
        for i in tqdm(range(num_steps), desc="Sampling"):
            t = 1.0 - i * dt
            x = self._euler_step(x, t, dt, class_label)
            trajectory.append(x.cpu().clone())
        
        return x, trajectory
