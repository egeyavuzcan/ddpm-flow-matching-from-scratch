"""
Unit tests for diffusion processes (DDPM and Flow Matching).
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from diffusion.ddpm.noise_schedule import NoiseSchedule
from diffusion.ddpm.forward_process import DDPMForwardProcess
from diffusion.flow_matching.probability_path import FlowMatchingPath
from diffusion.flow_matching.velocity_field import get_velocity_target, compute_flow_matching_loss


class TestDDPMForwardProcess:
    """Tests for DDPM forward process."""
    
    @pytest.fixture
    def process(self):
        schedule = NoiseSchedule(num_timesteps=1000)
        return DDPMForwardProcess(schedule)
    
    def test_add_noise_shape(self, process):
        """Verify output shape matches input."""
        x_0 = torch.randn(4, 3, 32, 32)
        t = torch.randint(0, 1000, (4,))
        
        x_t, noise = process.add_noise(x_0, t)
        
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape
    
    def test_low_noise_at_t0(self, process):
        """At t=0, x_t should be very close to x_0 (minimal noise)."""
        x_0 = torch.randn(4, 3, 32, 32)
        t = torch.zeros(4, dtype=torch.long)
        
        x_t, noise = process.add_noise(x_0, t)
        
        # At t=0, sqrt_alpha_bar ≈ 0.9999, sqrt_one_minus ≈ 0.01
        # So x_t ≈ 0.9999*x_0 + 0.01*noise, very close to x_0
        # Check that difference is small relative to data + noise
        diff = (x_t - x_0).abs().mean()
        noise_contribution = noise.abs().mean() * 0.02  # ~1-2% of noise
        
        assert diff < noise_contribution, f"At t=0, difference {diff:.4f} should be < {noise_contribution:.4f}"
    
    def test_mostly_noise_at_high_t(self, process):
        """At t=T-1, x_t should be mostly noise."""
        x_0 = torch.randn(4, 3, 32, 32)
        t = torch.full((4,), 999, dtype=torch.long)
        noise = torch.randn_like(x_0)
        
        x_t, _ = process.add_noise(x_0, t, noise=noise)
        
        # At t=999, x_t should be close to noise
        # sqrt_one_minus_alpha_bar ≈ 0.99, sqrt_alpha_bar ≈ 0.14
        correlation = torch.corrcoef(
            torch.stack([x_t.flatten(), noise.flatten()])
        )[0, 1]
        
        assert correlation > 0.9, f"Expected high correlation with noise, got {correlation}"
    
    def test_deterministic_with_fixed_noise(self, process):
        """Verify deterministic output with fixed noise."""
        x_0 = torch.randn(4, 3, 32, 32)
        t = torch.randint(0, 1000, (4,))
        noise = torch.randn_like(x_0)
        
        x_t1, _ = process.add_noise(x_0, t, noise=noise)
        x_t2, _ = process.add_noise(x_0, t, noise=noise)
        
        assert torch.allclose(x_t1, x_t2)
    
    def test_sample_timesteps_range(self, process):
        """Verify sampled timesteps are in valid range."""
        t = process.sample_timesteps(100, device=torch.device("cpu"))
        
        assert t.min() >= 0
        assert t.max() < 1000


class TestFlowMatchingPath:
    """Tests for Flow Matching probability path."""
    
    @pytest.fixture
    def path(self):
        return FlowMatchingPath()
    
    def test_interpolate_at_t0(self, path):
        """At t=0, x_t should equal x_0."""
        x_0 = torch.randn(4, 3, 32, 32)
        x_1 = torch.randn(4, 3, 32, 32)
        t = torch.zeros(4)
        
        x_t = path.interpolate(x_0, x_1, t)
        
        assert torch.allclose(x_t, x_0)
    
    def test_interpolate_at_t1(self, path):
        """At t=1, x_t should equal x_1."""
        x_0 = torch.randn(4, 3, 32, 32)
        x_1 = torch.randn(4, 3, 32, 32)
        t = torch.ones(4)
        
        x_t = path.interpolate(x_0, x_1, t)
        
        assert torch.allclose(x_t, x_1)
    
    def test_interpolate_at_t05(self, path):
        """At t=0.5, x_t should be midpoint."""
        x_0 = torch.randn(4, 3, 32, 32)
        x_1 = torch.randn(4, 3, 32, 32)
        t = torch.full((4,), 0.5)
        
        x_t = path.interpolate(x_0, x_1, t)
        expected = 0.5 * x_0 + 0.5 * x_1
        
        assert torch.allclose(x_t, expected)
    
    def test_get_noisy_sample_shape(self, path):
        """Verify output shape."""
        x_0 = torch.randn(4, 3, 32, 32)
        t = torch.rand(4)
        
        x_t, noise = path.get_noisy_sample(x_0, t)
        
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape
    
    def test_sample_timesteps_range(self, path):
        """Verify sampled timesteps are in [0, 1]."""
        t = path.sample_timesteps(100, device=torch.device("cpu"))
        
        assert t.min() >= 0.0
        assert t.max() <= 1.0


class TestVelocityField:
    """Tests for Flow Matching velocity field."""
    
    def test_velocity_target_shape(self):
        """Verify velocity target shape."""
        x_0 = torch.randn(4, 3, 32, 32)
        x_1 = torch.randn(4, 3, 32, 32)
        
        velocity = get_velocity_target(x_0, x_1)
        
        assert velocity.shape == x_0.shape
    
    def test_velocity_is_difference(self):
        """Verify velocity is x_1 - x_0."""
        x_0 = torch.randn(4, 3, 32, 32)
        x_1 = torch.randn(4, 3, 32, 32)
        
        velocity = get_velocity_target(x_0, x_1)
        expected = x_1 - x_0
        
        assert torch.allclose(velocity, expected)
    
    def test_flow_matching_loss(self):
        """Verify Flow Matching loss computation."""
        x_0 = torch.randn(4, 3, 32, 32)
        x_1 = torch.randn(4, 3, 32, 32)
        
        # Perfect prediction
        v_pred = x_1 - x_0
        loss = compute_flow_matching_loss(v_pred, x_0, x_1)
        
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    
    def test_flow_matching_loss_nonzero(self):
        """Verify loss is nonzero for wrong predictions."""
        x_0 = torch.randn(4, 3, 32, 32)
        x_1 = torch.randn(4, 3, 32, 32)
        
        # Wrong prediction
        v_pred = torch.randn_like(x_0)
        loss = compute_flow_matching_loss(v_pred, x_0, x_1)
        
        assert loss > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
