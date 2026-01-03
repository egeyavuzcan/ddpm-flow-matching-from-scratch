"""
Model factory for creating diffusion models.
"""
from typing import Optional, Dict, Any
import torch.nn as nn

from .unet.unet import UNet, UNetSmall


def create_model(
    model_type: str = "unet",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ("unet", "unet_small", "dit" in future)
        **kwargs: Model-specific arguments
    
    Returns:
        Initialized model
    
    Example:
        model = create_model("unet", num_classes=10, image_size=32)
    """
    model_registry = {
        "unet": UNet,
        "unet_small": UNetSmall,
        # "dit": DiT,  # Coming in Phase 2
    }
    
    if model_type not in model_registry:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(model_registry.keys())}"
        )
    
    model_class = model_registry[model_type]
    return model_class(**kwargs)


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a model type.
    
    Args:
        model_type: Type of model
    
    Returns:
        Dictionary of default config values
    """
    configs = {
        "unet": {
            "in_channels": 3,
            "out_channels": 3,
            "base_channels": 64,
            "channel_mults": (1, 2, 4, 8),
            "num_res_blocks": 2,
            "attention_resolutions": (16, 8),
            "time_emb_dim": 256,
            "num_classes": 10,
            "dropout": 0.1,
            "image_size": 32,
        },
        "unet_small": {
            "in_channels": 3,
            "out_channels": 3,
            "num_classes": 10,
            "image_size": 32,
        },
    }
    
    if model_type not in configs:
        raise ValueError(f"No config for model type: {model_type}")
    
    return configs[model_type]
