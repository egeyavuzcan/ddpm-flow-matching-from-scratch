"""
Patch Embedding for DiT (Diffusion Transformer).

Converts images to sequence of patches, similar to ViT.
"""
import torch
import torch.nn as nn
from torch import Tensor


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    
    Splits image into non-overlapping patches and projects to hidden dimension.
    
    Args:
        img_size: Input image size (assumed square)
        patch_size: Size of each patch (assumed square)
        in_channels: Number of input channels (3 for RGB)
        embed_dim: Embedding dimension
    
    Input: (B, C, H, W)
    Output: (B, num_patches, embed_dim)
    """
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolution with kernel=patch_size, stride=patch_size
        # Acts as patch extraction + linear projection
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size {H}x{W} doesn't match model {self.img_size}x{self.img_size}"
        
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        
        # (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x


class Unpatchify(nn.Module):
    """
    Convert patch sequence back to image.
    
    Args:
        img_size: Output image size
        patch_size: Size of each patch
        out_channels: Number of output channels (3 for RGB)
        embed_dim: Embedding dimension
    
    Input: (B, num_patches, patch_dim)
    Output: (B, C, H, W)
    """
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        out_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_patches = (img_size // patch_size) ** 2
        
        # Project to patch size
        patch_dim = out_channels * patch_size * patch_size
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, num_patches, patch_dim) where patch_dim = C*P*P
        
        Returns:
            (B, C, H, W)
        """
        B, N, patch_dim = x.shape
        assert N == self.num_patches, f"Expected {self.num_patches} patches, got {N}"
        
        # x is already (B, num_patches, C*P*P) from final layer
        # No projection needed
        
        # Reshape to image
        p = self.patch_size
        h = w = self.img_size // p
        
        # (B, N, C*P*P) -> (B, h, w, C, P, P)
        x = x.reshape(B, h, w, self.out_channels, p, p)
        
        # (B, h, w, C, P, P) -> (B, C, h, P, w, P)
        x = x.permute(0, 3, 1, 4, 2, 5)
        
        # (B, C, h, P, w, P) -> (B, C, h*P, w*P)
        x = x.reshape(B, self.out_channels, h * p, w * p)
        
        return x
