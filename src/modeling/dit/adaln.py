"""
Adaptive Layer Normalization (AdaLN) for DiT.

Uses timestep and class conditioning to modulate layer norm parameters.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero initialization.
    
    Introduced in DiT paper. Uses timestep and class embeddings to produce
    scale and shift parameters for layer normalization.
    
    Args:
        embed_dim: Model dimension
        num_classes: Number of classes (0 for unconditional)
    """
    
    def __init__(self, embed_dim: int, num_classes: int = 10):
        super().__init__()
        
        # Total conditioning dimension (time + class)
        # Time embedding: embed_dim
        # Class embedding: embed_dim (if num_classes > 0)
        cond_dim = embed_dim
        if num_classes > 0:
            cond_dim += embed_dim
        
        # MLP to produce modulation parameters
        # Outputs: (scale_1, shift_1, scale_2, shift_2, gate_1, gate_2)
        # 6 parameters per block (2 for attention, 2 for MLP, 2 gates)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * embed_dim, bias=True)
        )
        
        # Initialize to zero (important for stable training)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(
        self,
        time_emb: Tensor,
        class_emb: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            time_emb: (B, D) time embeddings
            class_emb: (B, D) class embeddings (optional)
        
        Returns:
            (scale_1, shift_1, gate_1, scale_2, shift_2, gate_2)
            Each: (B, D)
        """
        # Concatenate embeddings
        if class_emb is not None:
            c = torch.cat([time_emb, class_emb], dim=1)
        else:
            c = time_emb
        
        # (B, cond_dim) -> (B, 6*D)
        modulation = self.adaLN_modulation(c)
        
        # Split into 6 components
        # (B, 6*D) -> 6 x (B, D)
        scale_1, shift_1, gate_1, scale_2, shift_2, gate_2 = modulation.chunk(6, dim=1)
        
        return scale_1, shift_1, gate_1, scale_2, shift_2, gate_2


def modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    """
    Apply scale and shift modulation.
    
    Args:
        x: (B, N, D) input
        scale: (B, D) scale parameter
        shift: (B, D) shift parameter
    
    Returns:
        (B, N, D) modulated input
    """
    # (B, D) -> (B, 1, D) for broadcasting
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
