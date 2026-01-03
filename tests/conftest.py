"""
Pytest configuration and fixtures.
"""
import pytest
import torch
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def device():
    """Get available device (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def seed():
    """Default seed for reproducibility."""
    return 42


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 4


@pytest.fixture
def image_size():
    """Default image size (CIFAR-10)."""
    return 32


@pytest.fixture
def channels():
    """Default number of channels (RGB)."""
    return 3


@pytest.fixture
def timesteps():
    """Default number of timesteps for DDPM."""
    return 1000
