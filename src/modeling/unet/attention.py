"""
Attention modules for UNet.

Contains:
- SelfAttention: Multi-head self-attention for spatial features
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


class SelfAttention(nn.Module):
    """
    Multi-head self-attention for 2D feature maps.
    
    Typically used at low resolutions (e.g., 16x16, 8x8) in the UNet
    to capture global dependencies.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: int = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            channels: Number of input channels
            num_heads: Number of attention heads
            head_dim: Dimension per head (defaults to channels // num_heads)
            dropout: Dropout probability
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim or channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.to_out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply self-attention to spatial features.
        
        Args:
            x: (B, C, H, W) input features
        
        Returns:
            (B, C, H, W) output features (same shape as input)
        """
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention: (B, heads, head_dim, H*W)
        q = rearrange(q, 'b (h d) y x -> b h (y x) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) y x -> b h (y x) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) y x -> b h (y x) d', h=self.num_heads)
        
        # Attention: (B, heads, H*W, H*W)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape back: (B, C, H, W)
        out = rearrange(out, 'b h (y x) d -> b (h d) y x', y=H, x=W)
        
        # Output projection + residual
        out = self.to_out(out)
        
        return x + out


class LinearAttention(nn.Module):
    """
    Linear attention with O(N) complexity instead of O(N²).
    
    Useful for higher resolution feature maps where full attention
    would be too expensive.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.to_out = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = rearrange(q, 'b (h d) y x -> b h d (y x)', h=self.num_heads)
        k = rearrange(k, 'b (h d) y x -> b h d (y x)', h=self.num_heads)
        v = rearrange(v, 'b (h d) y x -> b h d (y x)', h=self.num_heads)
        
        # Apply softmax to q and k for linear attention
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)
        
        # Linear attention: O(N) instead of O(N²)
        context = torch.matmul(k, v.transpose(-1, -2))  # (B, h, d, d)
        out = torch.matmul(context.transpose(-1, -2), q)  # (B, h, d, N)
        
        out = rearrange(out, 'b h d (y x) -> b (h d) y x', y=H, x=W)
        out = self.to_out(out)
        
        return x + out


class AttentionBlock(nn.Module):
    """
    Wrapper that applies attention within a UNet block structure.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        use_linear_attention: bool = False,
    ):
        super().__init__()
        
        if use_linear_attention:
            self.attn = LinearAttention(channels, num_heads)
        else:
            self.attn = SelfAttention(channels, num_heads)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.attn(x)
