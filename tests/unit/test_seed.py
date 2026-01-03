"""
Unit tests for seed utility.
"""
import pytest
import torch
import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.seed import set_seed, get_generator

# NumPy is optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TestSetSeed:
    """Tests for set_seed function."""
    
    def test_torch_reproducibility(self):
        """Verify that torch random values are reproducible with same seed."""
        set_seed(42)
        tensor1 = torch.randn(10)
        
        set_seed(42)
        tensor2 = torch.randn(10)
        
        assert torch.allclose(tensor1, tensor2), "Torch tensors should be identical with same seed"
    
    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
    def test_numpy_reproducibility(self):
        """Verify that numpy random values are reproducible with same seed."""
        set_seed(42)
        array1 = np.random.randn(10)
        
        set_seed(42)
        array2 = np.random.randn(10)
        
        assert np.allclose(array1, array2), "Numpy arrays should be identical with same seed"
    
    def test_python_random_reproducibility(self):
        """Verify that Python random values are reproducible with same seed."""
        set_seed(42)
        values1 = [random.random() for _ in range(10)]
        
        set_seed(42)
        values2 = [random.random() for _ in range(10)]
        
        assert values1 == values2, "Python random values should be identical with same seed"
    
    def test_different_seeds_produce_different_results(self):
        """Verify that different seeds produce different random values."""
        set_seed(42)
        tensor1 = torch.randn(10)
        
        set_seed(123)
        tensor2 = torch.randn(10)
        
        assert not torch.allclose(tensor1, tensor2), "Different seeds should produce different tensors"


class TestGetGenerator:
    """Tests for get_generator function."""
    
    def test_generator_reproducibility(self):
        """Verify that generator produces reproducible values."""
        gen1 = get_generator(42)
        values1 = torch.randn(10, generator=gen1)
        
        gen2 = get_generator(42)
        values2 = torch.randn(10, generator=gen2)
        
        assert torch.allclose(values1, values2), "Generators with same seed should produce identical values"
    
    def test_generator_different_seeds(self):
        """Verify that different seeds produce different values."""
        gen1 = get_generator(42)
        values1 = torch.randn(10, generator=gen1)
        
        gen2 = get_generator(123)
        values2 = torch.randn(10, generator=gen2)
        
        assert not torch.allclose(values1, values2), "Different seeds should produce different values"
    
    def test_generator_returns_correct_type(self):
        """Verify that get_generator returns a torch.Generator."""
        gen = get_generator(42)
        assert isinstance(gen, torch.Generator), "Should return torch.Generator instance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
