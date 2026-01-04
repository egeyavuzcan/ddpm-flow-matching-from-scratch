"""
Unit tests for DiT (Diffusion Transformer).
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from modeling.dit.patch_embed import PatchEmbed, Unpatchify
from modeling.dit.adaln import AdaLNZero, modulate
from modeling.dit.dit_block import DiTBlock
from modeling.dit.dit import DiT, DiT_S, DiT_B
from modeling.model_factory import create_model


class TestPatchEmbed:
    """Tests for patch embedding."""
    
    def test_patch_embed_shape(self):
        """Test patch embedding output shape."""
        patch_embed = PatchEmbed(img_size=32, patch_size=4, embed_dim=384)
        
        x = torch.randn(2, 3, 32, 32)
        out = patch_embed(x)
        
        # (32/4)^2 = 64 patches
        assert out.shape == (2, 64, 384)
    
    def test_unpatchify_shape(self):
        """Test unpatchify output shape."""
        unpatchify = Unpatchify(img_size=32, patch_size=4, embed_dim=384)
        
        # unpatchify expects (B, N, C*P*P) = (2, 64, 48)
        x = torch.randn(2, 64, 3*4*4)  # 64 patches, patch_dim=48
        out = unpatchify(x)
        
        assert out.shape == (2, 3, 32, 32)
    
    def test_patch_unpatch_dimensions(self):
        """Test patch -> unpatch preserves dimensions (with manual projection)."""
        patch_embed = PatchEmbed(img_size=32, patch_size=4, embed_dim=384)
        unpatchify = Unpatchify(img_size=32, patch_size=4, embed_dim=384)
        
        x_in = torch.randn(2, 3, 32, 32)
        patches = patch_embed(x_in)  # (2, 64, 384)
        
        # Manually project to patch_dim for unpatchify
        # unpatchify expects (B, N, C*P*P) = (2, 64, 48)
        proj = torch.nn.Linear(384, 3*4*4)
        patches_proj = proj(patches)
        
        x_out = unpatchify(patches_proj)
        
        assert x_out.shape == x_in.shape


class TestAdaLN:
    """Tests for Adaptive Layer Normalization."""
    
    def test_adaln_output_shape(self):
        """Test AdaLN output shapes."""
        adaln = AdaLNZero(embed_dim=384, num_classes=10)
        
        time_emb = torch.randn(4, 384)
        class_emb = torch.randn(4, 384)
        
        params = adaln(time_emb, class_emb)
        
        # Should return 6 parameters
        assert len(params) == 6
        # Each should be (B, D)
        for param in params:
            assert param.shape == (4, 384)
    
    def test_modulate(self):
        """Test modulation function."""
        x = torch.randn(4, 64, 384)
        scale = torch.randn(4, 384)
        shift = torch.randn(4, 384)
        
        out = modulate(x, scale, shift)
        
        assert out.shape == x.shape


class TestDiTBlock:
    """Tests for DiT block."""
    
    def test_dit_block_forward(self):
        """Test DiT block forward pass."""
        block = DiTBlock(hidden_dim=384, num_heads=6)
        
        x = torch.randn(4, 64, 384)
        
        # Modulation parameters
        scale_1 = torch.randn(4, 384)
        shift_1 = torch.randn(4, 384)
        gate_1 = torch.randn(4, 384)
        scale_2 = torch.randn(4, 384)
        shift_2 = torch.randn(4, 384)
        gate_2 = torch.randn(4, 384)
        
        out = block(x, scale_1, shift_1, gate_1, scale_2, shift_2, gate_2)
        
        assert out.shape == x.shape


class TestDiT:
    """Tests for complete DiT model."""
    
    def test_dit_s_forward(self):
        """Test DiT-S forward pass."""
        model = DiT_S(img_size=32, num_classes=10)
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        y = torch.randint(0, 10, (2,))
        
        out = model(x, t, y)
        
        assert out.shape == (2, 3, 32, 32)
    
    def test_dit_b_forward(self):
        """Test DiT-B forward pass."""
        model = DiT_B(img_size=32, num_classes=10)
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        y = torch.randint(0, 10, (2,))
        
        out = model(x, t, y)
        
        assert out.shape == (2, 3, 32, 32)
    
    def test_dit_unconditional(self):
        """Test DiT without class conditioning."""
        model = DiT_S(img_size=32, num_classes=0)
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        
        out = model(x, t, y=None)
        
        assert out.shape == (2, 3, 32, 32)
    
    def test_dit_param_count(self):
        """Test DiT parameter count is reasonable."""
        model = DiT_S(img_size=32, num_classes=10)
        
        num_params = sum(p.numel() for p in model.parameters())
        
        # DiT-S should be ~23M params (smaller than original because of smaller image size)
        assert 20_000_000 < num_params < 30_000_000, f"Got {num_params:,} params"
    
    def test_model_factory(self):
        """Test creating DiT through model factory."""
        model = create_model("dit_s", img_size=32, num_classes=10)
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        y = torch.randint(0, 10, (2,))
        
        out = model(x, t, y)
        
        assert out.shape == (2, 3, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
