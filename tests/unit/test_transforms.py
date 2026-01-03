"""
Unit tests for dataset transforms.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dataset.transforms import normalize, denormalize, get_transform, get_inference_transform


class TestNormalize:
    """Tests for normalize function."""
    
    def test_normalize_range(self):
        """Verify normalize converts [0, 1] to [-1, 1]."""
        # Input in [0, 1]
        x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        result = normalize(x)
        expected = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
    
    def test_normalize_zero(self):
        """Verify 0 maps to -1."""
        x = torch.tensor([0.0])
        assert normalize(x).item() == -1.0
    
    def test_normalize_one(self):
        """Verify 1 maps to 1."""
        x = torch.tensor([1.0])
        assert normalize(x).item() == 1.0
    
    def test_normalize_half(self):
        """Verify 0.5 maps to 0."""
        x = torch.tensor([0.5])
        assert normalize(x).item() == 0.0
    
    def test_normalize_preserves_shape(self):
        """Verify normalize preserves tensor shape."""
        x = torch.rand(4, 3, 32, 32)
        result = normalize(x)
        assert result.shape == x.shape


class TestDenormalize:
    """Tests for denormalize function."""
    
    def test_denormalize_range(self):
        """Verify denormalize converts [-1, 1] to [0, 1]."""
        x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        result = denormalize(x)
        expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        
        assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
    
    def test_denormalize_clamps_overflow(self):
        """Verify denormalize clamps values outside [-1, 1]."""
        x = torch.tensor([-2.0, 2.0])  # Out of expected range
        result = denormalize(x)
        
        assert result[0].item() == 0.0, "Values below -1 should clamp to 0"
        assert result[1].item() == 1.0, "Values above 1 should clamp to 1"
    
    def test_denormalize_preserves_shape(self):
        """Verify denormalize preserves tensor shape."""
        x = torch.randn(4, 3, 32, 32)  # Random in roughly [-3, 3]
        result = denormalize(x)
        assert result.shape == x.shape


class TestRoundTrip:
    """Tests for normalize-denormalize round trip."""
    
    def test_round_trip_consistency(self):
        """Verify normalize -> denormalize returns original values."""
        original = torch.rand(4, 3, 32, 32)  # [0, 1]
        
        normalized = normalize(original)
        recovered = denormalize(normalized)
        
        assert torch.allclose(original, recovered, atol=1e-6), \
            "Round trip should recover original values"
    
    def test_inverse_round_trip(self):
        """Verify denormalize -> normalize returns original values."""
        # Create values in [-1, 1]
        original = torch.rand(4, 3, 32, 32) * 2 - 1
        
        denormalized = denormalize(original)
        recovered = normalize(denormalized)
        
        # Note: clamping may affect extreme values
        mask = (original >= -1) & (original <= 1)
        assert torch.allclose(original[mask], recovered[mask], atol=1e-6)


class TestGetTransform:
    """Tests for transform pipelines."""
    
    def test_get_transform_returns_compose(self):
        """Verify get_transform returns a Compose object."""
        from torchvision.transforms import Compose
        transform = get_transform(image_size=32)
        assert isinstance(transform, Compose)
    
    def test_get_inference_transform_returns_compose(self):
        """Verify get_inference_transform returns a Compose object."""
        from torchvision.transforms import Compose
        transform = get_inference_transform(image_size=32)
        assert isinstance(transform, Compose)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
