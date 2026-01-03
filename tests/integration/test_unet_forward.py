"""
Integration tests for complete UNet model.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from modeling.unet.unet import UNet, UNetSmall, count_parameters
from modeling.model_factory import create_model


class TestUNetForward:
    """Tests for UNet forward pass."""
    
    def test_unconditional_forward(self):
        """Test forward pass without class conditioning."""
        model = UNet(
            num_classes=None,
            base_channels=32,
            channel_mults=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            image_size=32,
        )
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        
        out = model(x, t)
        
        assert out.shape == (2, 3, 32, 32), f"Expected (2, 3, 32, 32), got {out.shape}"
    
    def test_conditional_forward(self):
        """Test forward pass with class conditioning."""
        model = UNet(
            num_classes=10,
            base_channels=32,
            channel_mults=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            image_size=32,
        )
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        c = torch.randint(0, 10, (2,))
        
        out = model(x, t, class_label=c)
        
        assert out.shape == (2, 3, 32, 32)
    
    def test_class_conditioning_affects_output(self):
        """Verify class conditioning changes the output."""
        model = UNet(
            num_classes=10,
            base_channels=32,
            channel_mults=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            image_size=32,
        )
        model.eval()
        
        torch.manual_seed(42)
        x = torch.randn(1, 3, 32, 32)
        t = torch.tensor([500])
        
        with torch.no_grad():
            out1 = model(x, t, class_label=torch.tensor([0]))
            out2 = model(x, t, class_label=torch.tensor([5]))
        
        assert not torch.allclose(out1, out2), "Different classes should produce different outputs"
    
    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        model = UNet(
            num_classes=10,
            base_channels=32,
            channel_mults=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            image_size=32,
        )
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 3, 32, 32)
            t = torch.randint(0, 1000, (batch_size,))
            c = torch.randint(0, 10, (batch_size,))
            
            out = model(x, t, c)
            assert out.shape == (batch_size, 3, 32, 32)


class TestUNetSmall:
    """Tests for UNetSmall variant."""
    
    def test_forward(self):
        """Test UNetSmall forward pass."""
        model = UNetSmall(num_classes=10)
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        c = torch.randint(0, 10, (2,))
        
        out = model(x, t, c)
        assert out.shape == (2, 3, 32, 32)
    
    def test_parameter_count(self):
        """Verify UNetSmall is actually small."""
        model = UNetSmall(num_classes=10)
        params = count_parameters(model)
        
        # Should be less than 5M parameters
        assert params < 5_000_000, f"UNetSmall has {params:,} params, expected < 5M"
        print(f"UNetSmall parameters: {params:,}")


class TestGradientFlow:
    """Tests for gradient flow through the model."""
    
    def test_backward_pass(self):
        """Verify gradients flow through the model."""
        model = UNet(
            num_classes=10,
            base_channels=32,
            channel_mults=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            image_size=32,
        )
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        t = torch.randint(0, 1000, (2,))
        c = torch.randint(0, 10, (2,))
        
        out = model(x, t, c)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_all_parameters_have_gradients(self):
        """Verify all parameters receive gradients."""
        model = UNet(
            num_classes=10,
            base_channels=32,
            channel_mults=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
            image_size=32,
        )
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        c = torch.randint(0, 10, (2,))
        
        out = model(x, t, c)
        loss = out.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestModelFactory:
    """Tests for model factory."""
    
    def test_create_unet(self):
        """Test creating UNet through factory."""
        model = create_model("unet", num_classes=10, image_size=32)
        assert isinstance(model, UNet)
    
    def test_create_unet_small(self):
        """Test creating UNetSmall through factory."""
        model = create_model("unet_small", num_classes=10)
        assert isinstance(model, UNetSmall)
    
    def test_invalid_model_type(self):
        """Test error on invalid model type."""
        with pytest.raises(ValueError):
            create_model("invalid_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
