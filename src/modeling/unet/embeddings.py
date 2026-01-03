"""
Embeddings for diffusion models.

Contains:
- SinusoidalPositionEmbedding: Time step embedding using sinusoidal functions
- ClassEmbedding: Learned class label embedding
- CombinedEmbedding: Time + Class combined embedding
"""
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for timesteps.
    
    Originally from "Attention Is All You Need" (Vaswani et al., 2017).
    Used in DDPM to encode the timestep information.
    
    For a timestep t and dimension i:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        """
        Args:
            dim: Embedding dimension (must be even)
            max_period: Maximum period for sinusoidal functions
        """
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, timesteps: Tensor) -> Tensor:
        """
        Create sinusoidal embeddings for timesteps.
        
        Args:
            timesteps: (B,) tensor of timestep indices (integer or float)
        
        Returns:
            (B, dim) tensor of sinusoidal embeddings
        """
        half_dim = self.dim // 2
        
        # Compute frequency bands
        # freqs = 1 / (10000^(2i/d)) = exp(-log(10000) * 2i/d)
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / half_dim
        )
        
        # Outer product: (B,) x (half_dim,) -> (B, half_dim)
        args = timesteps[:, None].float() * freqs[None, :]
        
        # Concatenate sin and cos: (B, half_dim) + (B, half_dim) -> (B, dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding


class ClassEmbedding(nn.Module):
    """
    Learned embedding for class labels.
    
    Maps integer class labels to dense vectors.
    """
    
    def __init__(self, num_classes: int, dim: int):
        """
        Args:
            num_classes: Number of classes (e.g., 10 for CIFAR-10)
            dim: Embedding dimension
        """
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.embedding = nn.Embedding(num_classes, dim)
    
    def forward(self, class_labels: Tensor) -> Tensor:
        """
        Embed class labels.
        
        Args:
            class_labels: (B,) tensor of class indices (0 to num_classes-1)
        
        Returns:
            (B, dim) tensor of class embeddings
        """
        return self.embedding(class_labels)


class CombinedEmbedding(nn.Module):
    """
    Combined time and class embedding module.
    
    Provides a unified interface for conditioning on both timestep and class.
    Time embedding is always applied; class embedding is optional.
    """
    
    def __init__(
        self, 
        time_dim: int, 
        num_classes: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        """
        Args:
            time_dim: Dimension for sinusoidal time embedding
            num_classes: Number of classes (None for unconditional)
            output_dim: Final output dimension after MLP projection
                       (defaults to time_dim if not specified)
        """
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.output_dim = output_dim or time_dim
        
        # Time embedding: sinusoidal -> MLP
        self.time_embed = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, self.output_dim),
            nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        
        # Class embedding (optional)
        if num_classes is not None:
            self.class_embed = ClassEmbedding(num_classes, self.output_dim)
        else:
            self.class_embed = None
    
    def forward(
        self, 
        timesteps: Tensor, 
        class_labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute combined embedding.
        
        Args:
            timesteps: (B,) tensor of timesteps
            class_labels: (B,) tensor of class labels (optional)
        
        Returns:
            (B, output_dim) tensor of combined embeddings
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)
        t_emb = self.time_mlp(t_emb)
        
        # Add class embedding if provided
        if class_labels is not None and self.class_embed is not None:
            c_emb = self.class_embed(class_labels)
            t_emb = t_emb + c_emb
        
        return t_emb


# Convenience function for creating embeddings
def get_timestep_embedding(
    timesteps: Tensor,
    dim: int,
    max_period: float = 10000.0,
) -> Tensor:
    """
    Functional interface for sinusoidal timestep embedding.
    
    Args:
        timesteps: (B,) tensor of timesteps
        dim: Embedding dimension
        max_period: Maximum period
    
    Returns:
        (B, dim) tensor of embeddings
    """
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * 
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) / half_dim
    )
    args = timesteps[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
