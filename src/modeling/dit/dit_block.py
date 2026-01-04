"""
DiT Block (Transformer Block with AdaLN conditioning).
"""
import torch
import torch.nn as nn
from torch import Tensor

from modeling.dit.adaln import modulate


class DiTBlock(nn.Module):
    """
    DiT Transformer Block with AdaLN-Zero conditioning.
    
    Architecture:
        1. LayerNorm + Self-Attention (with AdaLN modulation)
        2. LayerNorm + MLP (with AdaLN modulation)
        3. Gating (zero-initialized)
    
    Args:
        hidden_dim: Model dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim = hidden_dim * mlp_ratio
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Layer norms (without affine parameters, AdaLN will handle)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # MLP
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
    
    def forward(
        self,
        x: Tensor,
        scale_1: Tensor,
        shift_1: Tensor,
        gate_1: Tensor,
        scale_2: Tensor,
        shift_2: Tensor,
        gate_2: Tensor,
    ) -> Tensor:
        """
        Args:
            x: (B, N, D) input
            scale_1, shift_1, gate_1: (B, D) attention modulation
            scale_2, shift_2, gate_2: (B, D) MLP modulation
        
        Returns:
            (B, N, D) output
        """
        # Self-attention block
        # Modulate -> Norm -> Attention -> Gate -> Residual
        norm_x = self.norm1(x)
        mod_x = modulate(norm_x, scale_1, shift_1)
        
        # Self-attention (pytorch expects (B, N, D))
        attn_out, _ = self.attn(mod_x, mod_x, mod_x, need_weights=False)
        
        # Apply gate and residual
        x = x + gate_1.unsqueeze(1) * attn_out
        
        # MLP block
        # Modulate -> Norm -> MLP -> Gate -> Residual
        norm_x = self.norm2(x)
        mod_x = modulate(norm_x, scale_2, shift_2)
        
        mlp_out = self.mlp(mod_x)
        
        # Apply gate and residual
        x = x + gate_2.unsqueeze(1) * mlp_out
        
        return x
