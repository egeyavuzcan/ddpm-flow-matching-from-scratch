"""
Image transforms for diffusion models.

Diffusion models typically work with images normalized to [-1, 1] range.
"""
import torch
from torch import Tensor
from torchvision import transforms
from typing import Tuple


def normalize(x: Tensor) -> Tensor:
    """
    Normalize image from [0, 1] to [-1, 1] range.
    
    Args:
        x: Tensor in range [0, 1]
    
    Returns:
        Tensor in range [-1, 1]
    """
    return x * 2.0 - 1.0


def denormalize(x: Tensor) -> Tensor:
    """
    Denormalize image from [-1, 1] to [0, 1] range.
    
    Args:
        x: Tensor in range [-1, 1]
    
    Returns:
        Tensor in range [0, 1], clamped
    """
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def get_transform(
    image_size: int = 32,
    horizontal_flip: bool = True,
    to_tensor: bool = True,
) -> transforms.Compose:
    """
    Get standard transform pipeline for diffusion training.
    
    Pipeline:
    1. Resize to target size (if needed)
    2. Random horizontal flip (for augmentation)
    3. Convert to tensor [0, 1]
    4. Normalize to [-1, 1]
    
    Args:
        image_size: Target image size (assumes square)
        horizontal_flip: Apply random horizontal flip
        to_tensor: Convert to tensor (disable if already tensor)
    
    Returns:
        torchvision.transforms.Compose pipeline
    """
    transform_list = []
    
    # Resize if specified
    if image_size:
        transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Data augmentation
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    # Convert to tensor [0, 1]
    if to_tensor:
        transform_list.append(transforms.ToTensor())
    
    # Normalize to [-1, 1]
    transform_list.append(transforms.Lambda(normalize))
    
    return transforms.Compose(transform_list)


def get_inference_transform(image_size: int = 32) -> transforms.Compose:
    """
    Get transform pipeline for inference (no augmentation).
    
    Args:
        image_size: Target image size
    
    Returns:
        torchvision.transforms.Compose pipeline
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(normalize),
    ])
