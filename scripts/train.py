#!/usr/bin/env python
"""
Training script for DDPM and Flow Matching models.

Usage:
    python scripts/train.py --method ddpm --config configs/ddpm_cifar10.yaml
    python scripts/train.py --method flow_matching --config configs/flow_matching_cifar10.yaml
    
    # Quick test (1 epoch)
    python scripts/train.py --method ddpm --epochs 1 --batch_size 32
"""
import argparse
import sys
from pathlib import Path
import yaml
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset.cifar10 import get_cifar10_dataloader
from modeling.unet.unet import UNet, UNetSmall
from modeling.model_factory import create_model
from diffusion.ddpm.noise_schedule import NoiseSchedule
from training.ddpm_trainer import DDPMTrainer
from training.flow_matching_trainer import FlowMatchingTrainer
from utils.seed import set_seed


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Merge override config into base config."""
    config = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in config:
            config[key] = merge_configs(config[key], value)
        else:
            config[key] = value
    return config


def main():
    parser = argparse.ArgumentParser(description="Train diffusion models")
    parser.add_argument("--method", type=str, choices=["ddpm", "flow_matching"], 
                       default="ddpm", help="Training method")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (overrides config)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()
    
    # Load base config
    base_config_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    config = load_config(base_config_path)
    
    # Load method-specific config if provided
    if args.config:
        method_config = load_config(args.config)
        config = merge_configs(config, method_config)
    
    # Override with CLI arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["dataset"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.device:
        config["device"] = args.device
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
        config["paths"]["checkpoint_dir"] = f"{args.output_dir}/checkpoints"
        config["paths"]["log_dir"] = f"{args.output_dir}/logs"
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loader
    print("Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_dataloader(
        root=config["dataset"]["root"],
        train=True,
        batch_size=config["dataset"]["batch_size"],
        num_workers=config["dataset"]["num_workers"],
    )
    print(f"Dataset: {len(dataloader.dataset)} samples, {len(dataloader)} batches")
    
    # Create model
    print(f"Creating model: {config['model']['type']}")
    model = create_model(
        model_type=config["model"]["type"],
        num_classes=config["model"]["num_classes"],
        image_size=config["model"]["image_size"],
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Create output directories
    output_dir = Path(config["paths"]["output_dir"]) / args.method
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer based on method
    if args.method == "ddpm":
        print("Creating DDPM trainer...")
        schedule = NoiseSchedule(
            num_timesteps=config["ddpm"]["num_timesteps"],
            schedule_type=config["ddpm"]["schedule_type"],
            beta_start=config["ddpm"]["beta_start"],
            beta_end=config["ddpm"]["beta_end"],
        )
        trainer = DDPMTrainer(
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            log_dir=str(log_dir),
            checkpoint_dir=str(checkpoint_dir),
        )
    else:  # flow_matching
        print("Creating Flow Matching trainer...")
        trainer = FlowMatchingTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            log_dir=str(log_dir),
            checkpoint_dir=str(checkpoint_dir),
        )
    
    # Train
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")
    print("-" * 50)
    
    history = trainer.train(
        dataloader=dataloader,
        num_epochs=config["training"]["epochs"],
        save_every=config["training"]["save_every"],
        log_interval=config["training"]["log_interval"],
    )
    
    print("-" * 50)
    print(f"Training complete!")
    print(f"Final loss: {history['losses'][-1]:.4f}")
    print(f"Training time: {history['training_time']:.1f}s")
    print(f"Checkpoint saved: {checkpoint_dir / 'final.pt'}")
    
    # Log to TensorBoard
    print(f"\nTo view logs: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
