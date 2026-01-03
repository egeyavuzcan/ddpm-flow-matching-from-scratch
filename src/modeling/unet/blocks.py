"""
UNet building blocks for diffusion models.

Contains:
- ResidualBlock: Basic residual block with time/class conditioning
- DownBlock: Encoder block (ResBlocks + optional attention + downsample)
- UpBlock: Decoder block (ResBlocks + optional attention + upsample)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding conditioning.
    
    Architecture:
        x -> GroupNorm -> SiLU -> Conv -> (+t_emb) -> GroupNorm -> SiLU -> Dropout -> Conv -> (+skip)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
        num_groups: int = 32,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            time_emb_dim: Dimension of time embedding
            dropout: Dropout probability
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()
        
        # First convolution
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        # Second convolution
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection (if channels change)
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
    
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C_in, H, W) input features
            t_emb: (B, time_emb_dim) time embedding
        
        Returns:
            (B, C_out, H, W) output features
        """
        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        t = self.time_mlp(t_emb)
        h = h + t[:, :, None, None]  # Broadcast to spatial dims
        
        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip_conv(x)


class Downsample(nn.Module):
    """Spatial downsampling by factor of 2."""
    
    def __init__(self, channels: int, use_conv: bool = True):
        """
        Args:
            channels: Number of channels
            use_conv: If True, use strided conv. If False, use avg pool.
        """
        super().__init__()
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Spatial upsampling by factor of 2."""
    
    def __init__(self, channels: int, use_conv: bool = True):
        """
        Args:
            channels: Number of channels
            use_conv: If True, use conv after upsample.
        """
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class DownBlock(nn.Module):
    """
    Encoder block: multiple ResBlocks followed by downsampling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        downsample: bool = True,
    ):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            time_emb_dim: Time embedding dimension
            num_res_blocks: Number of residual blocks
            dropout: Dropout probability
            downsample: Whether to apply downsampling
        """
        super().__init__()
        
        self.res_blocks = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResidualBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                )
            )
            current_channels = out_channels
        
        self.downsample = Downsample(out_channels) if downsample else None
    
    def forward(
        self, 
        x: Tensor, 
        t_emb: Tensor,
    ) -> Tuple[Tensor, list]:
        """
        Forward pass.
        
        Args:
            x: (B, C_in, H, W) input features
            t_emb: (B, time_emb_dim) time embedding
        
        Returns:
            output: (B, C_out, H/2, W/2) if downsample else (B, C_out, H, W)
            skip_connections: List of intermediate features for skip connections
        """
        skip_connections = []
        
        for res_block in self.res_blocks:
            x = res_block(x, t_emb)
            skip_connections.append(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, skip_connections


class UpBlock(nn.Module):
    """
    Decoder block: upsample followed by multiple ResBlocks with skip connections.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        upsample: bool = True,
    ):
        """
        Args:
            in_channels: Input channels (should match encoder out_channels)
            out_channels: Output channels
            time_emb_dim: Time embedding dimension
            num_res_blocks: Number of residual blocks
            dropout: Dropout probability
            upsample: Whether to apply upsampling
        """
        super().__init__()
        
        self.upsample = Upsample(in_channels) if upsample else None
        
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            # First block takes concatenated skip connection
            res_in_channels = in_channels + in_channels if i == 0 else out_channels
            self.res_blocks.append(
                ResidualBlock(
                    in_channels=res_in_channels,
                    out_channels=out_channels,
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                )
            )
    
    def forward(
        self, 
        x: Tensor, 
        t_emb: Tensor,
        skip_connections: list,
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C_in, H, W) input features
            t_emb: (B, time_emb_dim) time embedding
            skip_connections: List of skip connection features from encoder
        
        Returns:
            (B, C_out, H*2, W*2) if upsample else (B, C_out, H, W)
        """
        if self.upsample is not None:
            x = self.upsample(x)
        
        for i, res_block in enumerate(self.res_blocks):
            if i == 0 and len(skip_connections) > 0:
                skip = skip_connections.pop()
                x = torch.cat([x, skip], dim=1)
            x = res_block(x, t_emb)
        
        return x


class MiddleBlock(nn.Module):
    """
    Middle block (bottleneck): ResBlock -> Optional Attention -> ResBlock
    """
    
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.res1 = ResidualBlock(
            in_channels=channels,
            out_channels=channels,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
        self.res2 = ResidualBlock(
            in_channels=channels,
            out_channels=channels,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )
    
    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x
