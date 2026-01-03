"""
Unit tests for DDPM noise schedule.
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from diffusion.ddpm.noise_schedule import (
    NoiseSchedule,
    linear_beta_schedule,
    cosine_beta_schedule,
)


class TestLinearBetaSchedule:
    """Tests for linear beta schedule."""
    
    def test_shape(self):
        """Verify output shape."""
        betas = linear_beta_schedule(num_timesteps=1000)
        assert betas.shape == (1000,)
    
    def test_monotonic_increasing(self):
        """Verify betas increase monotonically."""
        betas = linear_beta_schedule(num_timesteps=1000)
        assert torch.all(betas[1:] >= betas[:-1])
    
    def test_boundary_values(self):
        """Verify start and end values."""
        betas = linear_beta_schedule(
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
        )
        assert torch.isclose(betas[0], torch.tensor(0.0001))
        assert torch.isclose(betas[-1], torch.tensor(0.02))


class TestCosineBetaSchedule:
    """Tests for cosine beta schedule."""
    
    def test_shape(self):
        """Verify output shape."""
        betas = cosine_beta_schedule(num_timesteps=1000)
        assert betas.shape == (1000,)
    
    def test_bounded(self):
        """Verify betas are bounded."""
        betas = cosine_beta_schedule(num_timesteps=1000)
        assert torch.all(betas > 0)
        assert torch.all(betas < 1)


class TestNoiseSchedule:
    """Tests for NoiseSchedule class."""
    
    def test_alpha_bar_decreasing(self):
        """Verify alpha_bar decreases monotonically."""
        schedule = NoiseSchedule(num_timesteps=1000)
        assert torch.all(schedule.alpha_bar[1:] <= schedule.alpha_bar[:-1])
    
    def test_alpha_bar_boundaries(self):
        """Verify alpha_bar boundaries."""
        schedule = NoiseSchedule(num_timesteps=1000, schedule_type="linear")
        
        # At t=0, alpha_bar should be close to 1 (almost no noise)
        assert schedule.alpha_bar[0] > 0.99
        
        # At t=T-1, alpha_bar should be close to 0 (almost pure noise)
        assert schedule.alpha_bar[-1] < 0.05
    
    def test_sqrt_values_consistency(self):
        """Verify sqrt values are consistent with alpha_bar."""
        schedule = NoiseSchedule(num_timesteps=1000)
        
        reconstructed = schedule.sqrt_alpha_bar ** 2 + schedule.sqrt_one_minus_alpha_bar ** 2
        expected = torch.ones_like(reconstructed)
        
        assert torch.allclose(reconstructed, expected, atol=1e-5)
    
    def test_get_value_shape(self):
        """Verify get_value broadcasts correctly."""
        schedule = NoiseSchedule(num_timesteps=1000)
        
        t = torch.tensor([0, 500, 999])
        shape = (3, 3, 32, 32)
        
        values = schedule.get_value(schedule.sqrt_alpha_bar, t, shape)
        
        assert values.shape == (3, 1, 1, 1)
    
    def test_cosine_schedule_smoother(self):
        """Verify cosine schedule has smoother alpha_bar at low t."""
        linear = NoiseSchedule(num_timesteps=1000, schedule_type="linear")
        cosine = NoiseSchedule(num_timesteps=1000, schedule_type="cosine")
        
        # At low t, cosine should preserve more signal
        # (alpha_bar should be higher for cosine at early steps)
        assert cosine.alpha_bar[100] > linear.alpha_bar[100]


class TestNoiseScheduleDevice:
    """Tests for device handling."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_move_to_cuda(self):
        """Verify schedule can be moved to CUDA."""
        schedule = NoiseSchedule(num_timesteps=100)
        schedule = schedule.cuda()
        
        assert schedule.betas.device.type == "cuda"
        assert schedule.alpha_bar.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
