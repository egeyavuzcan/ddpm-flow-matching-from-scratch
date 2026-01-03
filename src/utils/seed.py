"""
Seed utility for reproducibility.
"""
import random
import torch

# NumPy is optional for seed setting
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    
    if HAS_NUMPY:
        np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_generator(seed: int = 42, device: str = "cpu") -> torch.Generator:
    """
    Create a torch Generator with a specific seed.
    
    Args:
        seed: Random seed value
        device: Device for the generator ('cpu' or 'cuda')
    
    Returns:
        torch.Generator with the specified seed
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
