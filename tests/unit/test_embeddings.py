"""
Unit tests for embedding modules.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from modeling.unet.embeddings import (
    SinusoidalPositionEmbedding,
    ClassEmbedding,
    CombinedEmbedding,
    get_timestep_embedding,
)


class TestSinusoidalPositionEmbedding:
    """Tests for SinusoidalPositionEmbedding."""
    
    def test_output_shape(self):
        """Verify output shape is (B, dim)."""
        embed = SinusoidalPositionEmbedding(dim=256)
        timesteps = torch.tensor([0, 100, 500, 999])
        output = embed(timesteps)
        
        assert output.shape == (4, 256), f"Expected (4, 256), got {output.shape}"
    
    def test_different_timesteps_different_embeddings(self):
        """Verify different timesteps produce different embeddings."""
        embed = SinusoidalPositionEmbedding(dim=256)
        
        t0 = embed(torch.tensor([0]))
        t500 = embed(torch.tensor([500]))
        t999 = embed(torch.tensor([999]))
        
        # All should be different
        assert not torch.allclose(t0, t500), "t=0 and t=500 should have different embeddings"
        assert not torch.allclose(t500, t999), "t=500 and t=999 should have different embeddings"
        assert not torch.allclose(t0, t999), "t=0 and t=999 should have different embeddings"
    
    def test_same_timestep_same_embedding(self):
        """Verify same timestep produces same embedding."""
        embed = SinusoidalPositionEmbedding(dim=256)
        
        t1 = embed(torch.tensor([500]))
        t2 = embed(torch.tensor([500]))
        
        assert torch.allclose(t1, t2), "Same timestep should produce same embedding"
    
    def test_dimension_must_be_even(self):
        """Verify odd dimension raises error."""
        with pytest.raises(AssertionError):
            SinusoidalPositionEmbedding(dim=255)
    
    def test_batch_dimension(self):
        """Verify batched input works correctly."""
        embed = SinusoidalPositionEmbedding(dim=128)
        batch = torch.randint(0, 1000, (32,))
        output = embed(batch)
        
        assert output.shape == (32, 128)
    
    def test_float_timesteps(self):
        """Verify float timesteps work (for flow matching)."""
        embed = SinusoidalPositionEmbedding(dim=256)
        timesteps = torch.tensor([0.0, 0.5, 1.0])
        output = embed(timesteps)
        
        assert output.shape == (3, 256)


class TestClassEmbedding:
    """Tests for ClassEmbedding."""
    
    def test_output_shape(self):
        """Verify output shape is (B, dim)."""
        embed = ClassEmbedding(num_classes=10, dim=256)
        labels = torch.tensor([0, 5, 9, 3])
        output = embed(labels)
        
        assert output.shape == (4, 256), f"Expected (4, 256), got {output.shape}"
    
    def test_different_classes_different_embeddings(self):
        """Verify different classes produce different embeddings."""
        embed = ClassEmbedding(num_classes=10, dim=256)
        
        c0 = embed(torch.tensor([0]))
        c5 = embed(torch.tensor([5]))
        c9 = embed(torch.tensor([9]))
        
        assert not torch.allclose(c0, c5), "Different classes should have different embeddings"
        assert not torch.allclose(c5, c9), "Different classes should have different embeddings"
    
    def test_same_class_same_embedding(self):
        """Verify same class produces same embedding."""
        embed = ClassEmbedding(num_classes=10, dim=256)
        
        c1 = embed(torch.tensor([5]))
        c2 = embed(torch.tensor([5]))
        
        assert torch.allclose(c1, c2), "Same class should produce same embedding"
    
    def test_all_classes_valid(self):
        """Verify all class indices work."""
        embed = ClassEmbedding(num_classes=10, dim=256)
        
        for i in range(10):
            output = embed(torch.tensor([i]))
            assert output.shape == (1, 256)


class TestCombinedEmbedding:
    """Tests for CombinedEmbedding."""
    
    def test_unconditional_mode(self):
        """Test without class conditioning."""
        embed = CombinedEmbedding(time_dim=256, num_classes=None)
        timesteps = torch.tensor([0, 500, 999])
        output = embed(timesteps)
        
        assert output.shape == (3, 256)
    
    def test_conditional_mode(self):
        """Test with class conditioning."""
        embed = CombinedEmbedding(time_dim=256, num_classes=10)
        timesteps = torch.tensor([0, 500, 999])
        labels = torch.tensor([0, 5, 9])
        output = embed(timesteps, labels)
        
        assert output.shape == (3, 256)
    
    def test_conditioning_changes_output(self):
        """Verify class conditioning affects output."""
        embed = CombinedEmbedding(time_dim=256, num_classes=10)
        timesteps = torch.tensor([500, 500, 500])
        
        # Same timestep, different classes
        labels1 = torch.tensor([0, 0, 0])
        labels2 = torch.tensor([5, 5, 5])
        
        out1 = embed(timesteps, labels1)
        out2 = embed(timesteps, labels2)
        
        assert not torch.allclose(out1, out2), "Different classes should change output"
    
    def test_custom_output_dim(self):
        """Test custom output dimension."""
        embed = CombinedEmbedding(time_dim=256, num_classes=10, output_dim=512)
        timesteps = torch.tensor([500])
        labels = torch.tensor([5])
        output = embed(timesteps, labels)
        
        assert output.shape == (1, 512)
    
    def test_class_labels_ignored_when_unconditional(self):
        """Verify class labels are ignored when num_classes=None."""
        embed = CombinedEmbedding(time_dim=256, num_classes=None)
        timesteps = torch.tensor([500])
        labels = torch.tensor([5])  # This should be ignored
        
        out_with_labels = embed(timesteps, labels)
        out_without_labels = embed(timesteps)
        
        assert torch.allclose(out_with_labels, out_without_labels)


class TestGetTimestepEmbedding:
    """Tests for functional get_timestep_embedding."""
    
    def test_output_shape(self):
        """Verify output shape."""
        timesteps = torch.tensor([0, 500, 999])
        output = get_timestep_embedding(timesteps, dim=256)
        
        assert output.shape == (3, 256)
    
    def test_matches_module(self):
        """Verify functional version matches module version."""
        embed_module = SinusoidalPositionEmbedding(dim=256)
        timesteps = torch.tensor([0, 500, 999])
        
        module_out = embed_module(timesteps)
        func_out = get_timestep_embedding(timesteps, dim=256)
        
        assert torch.allclose(module_out, func_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
