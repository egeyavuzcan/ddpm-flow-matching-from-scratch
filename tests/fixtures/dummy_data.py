"""
Dummy data generators for testing.
"""
import torch
from typing import Tuple


def create_dummy_images(
    batch_size: int = 4,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
    device: str = "cpu",
    normalized: bool = True,
) -> torch.Tensor:
    """
    Create dummy image tensors for testing.
    
    Args:
        batch_size: Number of images in batch
        channels: Number of channels (3 for RGB)
        height: Image height
        width: Image width
        device: Device to create tensor on
        normalized: If True, range is [-1, 1], else [0, 1]
    
    Returns:
        Tensor of shape (B, C, H, W)
    """
    images = torch.randn(batch_size, channels, height, width, device=device)
    if normalized:
        # Clamp to [-1, 1] range
        images = torch.clamp(images, -1, 1)
    else:
        # Range [0, 1]
        images = torch.sigmoid(images)
    return images


def create_dummy_timesteps(
    batch_size: int = 4,
    num_timesteps: int = 1000,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Create random timestep indices for testing.
    
    Args:
        batch_size: Number of timesteps in batch
        num_timesteps: Total number of timesteps (T)
        device: Device to create tensor on
    
    Returns:
        Tensor of shape (B,) with random integers in [0, T-1]
    """
    return torch.randint(0, num_timesteps, (batch_size,), device=device)


def create_dummy_noise(
    batch_size: int = 4,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Create standard Gaussian noise for testing.
    
    Args:
        batch_size: Number of noise tensors
        channels: Number of channels
        height: Image height
        width: Image width
        device: Device to create tensor on
    
    Returns:
        Tensor of shape (B, C, H, W) with N(0, 1) noise
    """
    return torch.randn(batch_size, channels, height, width, device=device)


def create_dummy_continuous_timesteps(
    batch_size: int = 4,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Create random continuous timesteps in [0, 1] for Flow Matching.
    
    Args:
        batch_size: Number of timesteps
        device: Device to create tensor on
    
    Returns:
        Tensor of shape (B,) with random floats in [0, 1]
    """
    return torch.rand(batch_size, device=device)
