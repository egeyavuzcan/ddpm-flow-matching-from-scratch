"""
Complete UNet architecture for diffusion models.

Class-conditional UNet with:
- Sinusoidal time embeddings
- Class embeddings (optional)
- Encoder-Decoder with skip connections
- Optional self-attention at specified resolutions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple

from .embeddings import CombinedEmbedding
from .blocks import ResidualBlock, Downsample, Upsample
from .attention import SelfAttention


class UNet(nn.Module):
    """
    UNet for diffusion models with class conditioning.
    
    Simplified architecture with clear skip connection tracking:
        - Encoder: [conv_in] -> levels of [ResBlocks + Downsample]
        - Middle: ResBlock -> Attention -> ResBlock  
        - Decoder: levels of [Upsample + ResBlocks (with skip concat)]
        - Output: norm -> conv_out
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        time_emb_dim: int = 256,
        num_classes: Optional[int] = 10,
        dropout: float = 0.1,
        image_size: int = 32,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_res_blocks = num_res_blocks
        
        num_levels = len(channel_mults)
        channels = [base_channels * m for m in channel_mults]
        
        # Time and class embedding
        self.time_class_embed = CombinedEmbedding(
            time_dim=time_emb_dim,
            num_classes=num_classes,
            output_dim=time_emb_dim,
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # ==== ENCODER ====
        self.encoder = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        skip_channels = [base_channels]  # Track channels for skip connections
        
        current_ch = base_channels
        current_res = image_size
        
        for level, ch in enumerate(channels):
            is_last = (level == num_levels - 1)
            
            for _ in range(num_res_blocks):
                self.encoder.append(ResidualBlock(current_ch, ch, time_emb_dim, dropout))
                current_ch = ch
                skip_channels.append(current_ch)
                
                if current_res in attention_resolutions:
                    self.encoder_attns.append(SelfAttention(current_ch))
                else:
                    self.encoder_attns.append(nn.Identity())
            
            if not is_last:
                self.encoder.append(Downsample(current_ch))
                skip_channels.append(current_ch)
                self.encoder_attns.append(nn.Identity())
                current_res //= 2
        
        # ==== MIDDLE ====
        mid_ch = channels[-1]
        self.mid_block1 = ResidualBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_block2 = ResidualBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        
        # ==== DECODER ====
        self.decoder = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        
        for level, ch in enumerate(reversed(channels)):
            is_last = (level == num_levels - 1)
            
            for i in range(num_res_blocks + 1):
                # Concatenate with skip connection
                skip_ch = skip_channels.pop()
                self.decoder.append(
                    ResidualBlock(current_ch + skip_ch, ch, time_emb_dim, dropout)
                )
                current_ch = ch
                
                if current_res in attention_resolutions:
                    self.decoder_attns.append(SelfAttention(current_ch))
                else:
                    self.decoder_attns.append(nn.Identity())
            
            if not is_last:
                self.decoder.append(Upsample(current_ch))
                self.decoder_attns.append(nn.Identity())
                current_res *= 2
        
        # Output
        self.norm_out = nn.GroupNorm(min(32, base_channels), base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        class_label: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) noisy image
            t: (B,) timesteps
            class_label: (B,) class labels (optional)
        
        Returns:
            (B, C, H, W) predicted noise or velocity
        """
        # Embeddings
        emb = self.time_class_embed(t, class_label)
        
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder
        skips = [h]
        for block, attn in zip(self.encoder, self.encoder_attns):
            if isinstance(block, ResidualBlock):
                h = block(h, emb)
            else:  # Downsample
                h = block(h)
            h = attn(h)
            skips.append(h)
        
        # Middle
        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)
        
        # Decoder
        for block, attn in zip(self.decoder, self.decoder_attns):
            if isinstance(block, ResidualBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, emb)
            else:  # Upsample
                h = block(h)
            h = attn(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


class UNetSmall(UNet):
    """
    Smaller UNet variant for quick experiments.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_classes: Optional[int] = 10,
        image_size: int = 32,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=32,
            channel_mults=(1, 2, 4),
            num_res_blocks=1,
            attention_resolutions=(8,),
            time_emb_dim=128,
            num_classes=num_classes,
            dropout=0.0,
            image_size=image_size,
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
