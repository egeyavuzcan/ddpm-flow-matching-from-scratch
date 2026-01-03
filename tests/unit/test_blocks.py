"""
Unit tests for UNet blocks.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from modeling.unet.blocks import (
    ResidualBlock,
    Downsample,
    Upsample,
    DownBlock,
    UpBlock,
    MiddleBlock,
)


class TestResidualBlock:
    """Tests for ResidualBlock."""
    
    def test_same_channels(self):
        """Test when in_channels == out_channels."""
        block = ResidualBlock(
            in_channels=64,
            out_channels=64,
            time_emb_dim=256,
        )
        
        x = torch.randn(4, 64, 32, 32)
        t_emb = torch.randn(4, 256)
        out = block(x, t_emb)
        
        assert out.shape == (4, 64, 32, 32)
    
    def test_different_channels(self):
        """Test when in_channels != out_channels."""
        block = ResidualBlock(
            in_channels=64,
            out_channels=128,
            time_emb_dim=256,
        )
        
        x = torch.randn(4, 64, 32, 32)
        t_emb = torch.randn(4, 256)
        out = block(x, t_emb)
        
        assert out.shape == (4, 128, 32, 32)
    
    def test_preserves_spatial(self):
        """Verify spatial dimensions are preserved."""
        block = ResidualBlock(
            in_channels=64,
            out_channels=128,
            time_emb_dim=256,
        )
        
        x = torch.randn(4, 64, 16, 16)
        t_emb = torch.randn(4, 256)
        out = block(x, t_emb)
        
        assert out.shape[2:] == (16, 16)
    
    def test_time_conditioning_affects_output(self):
        """Verify different time embeddings produce different outputs."""
        block = ResidualBlock(
            in_channels=64,
            out_channels=64,
            time_emb_dim=256,
        )
        
        x = torch.randn(4, 64, 32, 32)
        t_emb1 = torch.randn(4, 256)
        t_emb2 = torch.randn(4, 256)
        
        out1 = block(x, t_emb1)
        out2 = block(x, t_emb2)
        
        assert not torch.allclose(out1, out2)


class TestDownsample:
    """Tests for Downsample."""
    
    def test_halves_spatial(self):
        """Verify spatial dimensions are halved."""
        down = Downsample(channels=64)
        x = torch.randn(4, 64, 32, 32)
        out = down(x)
        
        assert out.shape == (4, 64, 16, 16)
    
    def test_preserves_channels(self):
        """Verify channels are preserved."""
        down = Downsample(channels=128)
        x = torch.randn(4, 128, 32, 32)
        out = down(x)
        
        assert out.shape[1] == 128


class TestUpsample:
    """Tests for Upsample."""
    
    def test_doubles_spatial(self):
        """Verify spatial dimensions are doubled."""
        up = Upsample(channels=64)
        x = torch.randn(4, 64, 16, 16)
        out = up(x)
        
        assert out.shape == (4, 64, 32, 32)
    
    def test_preserves_channels(self):
        """Verify channels are preserved."""
        up = Upsample(channels=128)
        x = torch.randn(4, 128, 16, 16)
        out = up(x)
        
        assert out.shape[1] == 128


class TestDownBlock:
    """Tests for DownBlock."""
    
    def test_output_shape_with_downsample(self):
        """Test output shape when downsampling is enabled."""
        block = DownBlock(
            in_channels=64,
            out_channels=128,
            time_emb_dim=256,
            num_res_blocks=2,
            downsample=True,
        )
        
        x = torch.randn(4, 64, 32, 32)
        t_emb = torch.randn(4, 256)
        out, skips = block(x, t_emb)
        
        assert out.shape == (4, 128, 16, 16)
        assert len(skips) == 2  # One skip per ResBlock
    
    def test_output_shape_without_downsample(self):
        """Test output shape when downsampling is disabled."""
        block = DownBlock(
            in_channels=64,
            out_channels=128,
            time_emb_dim=256,
            num_res_blocks=2,
            downsample=False,
        )
        
        x = torch.randn(4, 64, 32, 32)
        t_emb = torch.randn(4, 256)
        out, skips = block(x, t_emb)
        
        assert out.shape == (4, 128, 32, 32)
    
    def test_skip_connection_shapes(self):
        """Verify skip connections have correct shapes."""
        block = DownBlock(
            in_channels=64,
            out_channels=128,
            time_emb_dim=256,
            num_res_blocks=2,
            downsample=True,
        )
        
        x = torch.randn(4, 64, 32, 32)
        t_emb = torch.randn(4, 256)
        out, skips = block(x, t_emb)
        
        # All skips should have out_channels and original spatial size
        for skip in skips:
            assert skip.shape == (4, 128, 32, 32)


class TestUpBlock:
    """Tests for UpBlock."""
    
    def test_output_shape_with_upsample(self):
        """Test output shape when upsampling is enabled."""
        block = UpBlock(
            in_channels=128,
            out_channels=64,
            time_emb_dim=256,
            num_res_blocks=2,
            upsample=True,
        )
        
        x = torch.randn(4, 128, 16, 16)
        t_emb = torch.randn(4, 256)
        # Skip connections from encoder
        skips = [torch.randn(4, 128, 32, 32)]
        out = block(x, t_emb, skips)
        
        assert out.shape == (4, 64, 32, 32)
    
    def test_output_shape_without_upsample(self):
        """Test output shape when upsampling is disabled."""
        block = UpBlock(
            in_channels=128,
            out_channels=64,
            time_emb_dim=256,
            num_res_blocks=2,
            upsample=False,
        )
        
        x = torch.randn(4, 128, 16, 16)
        t_emb = torch.randn(4, 256)
        skips = [torch.randn(4, 128, 16, 16)]
        out = block(x, t_emb, skips)
        
        assert out.shape == (4, 64, 16, 16)


class TestMiddleBlock:
    """Tests for MiddleBlock."""
    
    def test_preserves_shape(self):
        """Verify middle block preserves shape."""
        block = MiddleBlock(
            channels=256,
            time_emb_dim=256,
        )
        
        x = torch.randn(4, 256, 8, 8)
        t_emb = torch.randn(4, 256)
        out = block(x, t_emb)
        
        assert out.shape == (4, 256, 8, 8)


class TestGradientFlow:
    """Test gradient flow through blocks."""
    
    def test_resblock_gradients(self):
        """Verify gradients flow through ResidualBlock."""
        block = ResidualBlock(64, 128, 256)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        t = torch.randn(2, 256)
        
        out = block(x, t)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
