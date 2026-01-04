"""
DiT (Diffusion Transformer) Model.

Complete implementation of DiT architecture.
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from modeling.dit.patch_embed import PatchEmbed, Unpatchify
from modeling.dit.adaln import AdaLNZero
from modeling.dit.dit_block import DiTBlock
from modeling.unet.embeddings import SinusoidalPositionEmbedding


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT).
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_channels: Number of input channels
        hidden_dim: Model dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        num_classes: Number of classes (0 for unconditional)
        learn_sigma: If True, predict both mean and variance
    """
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_patches = (img_size // patch_size) ** 2
        self.learn_sigma = learn_sigma
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_dim,
        )
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        
        # Time embedding (sinusoidal)
        self.time_embed = SinusoidalPositionEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Class embedding (if conditional)
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, hidden_dim)
        else:
            self.class_embed = None
        
        # AdaLN conditioning
        self.adaln = AdaLNZero(hidden_dim, num_classes)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            out_channels=self.out_channels,
        )
        
        # Unpatchify
        self.unpatchify = Unpatchify(
            img_size=img_size,
            patch_size=patch_size,
            out_channels=self.out_channels,
            embed_dim=hidden_dim,
        )
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights following ViT and DiT papers."""
        # Position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Linear layers
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        self.apply(_init_weights)
        
        # Zero-init final layer (for stable training)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)
    
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (B, C, H, W) input images
            t: (B,) timesteps
            y: (B,) class labels (optional)
        
        Returns:
            (B, C, H, W) predicted noise/velocity
        """
        B = x.shape[0]
        
        # 1. Patch embedding
        # (B, C, H, W) -> (B, N, D)
        x = self.patch_embed(x)
        
        # 2. Add position embeddings
        x = x + self.pos_embed
        
        # 3. Time embedding
        # (B,) -> (B, D)
        time_emb = self.time_embed(t)
        time_emb = self.time_mlp(time_emb)
        
        # 4. Class embedding (if provided)
        if y is not None and self.class_embed is not None:
            class_emb = self.class_embed(y)
        else:
            class_emb = None
        
        # 5. Get AdaLN modulation parameters
        scale_1, shift_1, gate_1, scale_2, shift_2, gate_2 = self.adaln(time_emb, class_emb)
        
        # 6. Transformer blocks
        for block in self.blocks:
            x = block(x, scale_1, shift_1, gate_1, scale_2, shift_2, gate_2)
        
        # 7. Final layer (with conditioning)
        x = self.final_layer(x, scale_1, shift_1)
        
        # 8. Unpatchify
        # (B, N, D) -> (B, C, H, W)
        x = self.unpatchify(x)
        
        return x


class FinalLayer(nn.Module):
    """
    Final layer of DiT with AdaLN modulation.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        patch_size: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)
    
    def forward(self, x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, D)
            scale: (B, D)
            shift: (B, D)
        
        Returns:
            (B, N, patch_dim)
        """
        from modeling.dit.adaln import modulate
        
        x = self.norm(x)
        x = modulate(x, scale, shift)
        x = self.linear(x)
        return x


# DiT model configurations
def DiT_S(img_size=32, num_classes=10, **kwargs):
    """DiT-S: Small model (23M params for 32x32)."""
    # Handle both img_size and image_size for compatibility
    if 'image_size' in kwargs:
        img_size = kwargs.pop('image_size')
    
    return DiT(
        img_size=img_size,
        patch_size=4,
        hidden_dim=384,
        depth=12,
        num_heads=6,
        num_classes=num_classes,
        **kwargs
    )


def DiT_B(img_size=32, num_classes=10, **kwargs):
    """DiT-B: Base model (100M params for 32x32)."""
    if 'image_size' in kwargs:
        img_size = kwargs.pop('image_size')
    
    return DiT(
        img_size=img_size,
        patch_size=4,
        hidden_dim=768,
        depth=12,
        num_heads=12,
        num_classes=num_classes,
        **kwargs
    )


def DiT_L(img_size=32, num_classes=10, **kwargs):
    """DiT-L: Large model (458M params)."""
    if 'image_size' in kwargs:
        img_size = kwargs.pop('image_size')
    
    return DiT(
        img_size=img_size,
        patch_size=4,
        hidden_dim=1024,
        depth=24,
        num_heads=16,
        num_classes=num_classes,
        **kwargs
    )


def DiT_XL(img_size=32, num_classes=10, **kwargs):
    """DiT-XL: XLarge model (675M params)."""
    if 'image_size' in kwargs:
        img_size = kwargs.pop('image_size')
    
    return DiT(
        img_size=img_size,
        patch_size=4,
        hidden_dim=1152,
        depth=28,
        num_heads=16,
        num_classes=num_classes,
        **kwargs
    )
