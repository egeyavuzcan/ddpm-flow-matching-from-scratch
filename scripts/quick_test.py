#!/usr/bin/env python
"""
Quick test script to validate the full pipeline.

Tests:
1. Train both DDPM and Flow Matching for a few steps
2. Save checkpoints
3. Load checkpoints
4. Generate samples

This runs on CPU by default and uses minimal data for fast testing.

Usage:
    python scripts/quick_test.py
    python scripts/quick_test.py --device cuda  # Use GPU if available
"""
import sys
from pathlib import Path
import tempfile
import torch
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torch.utils.data import DataLoader, TensorDataset
from modeling.unet.unet import UNetSmall
from diffusion.ddpm.noise_schedule import NoiseSchedule
from training.ddpm_trainer import DDPMTrainer
from training.flow_matching_trainer import FlowMatchingTrainer
from inference.ddpm_sampler import DDPMSampler
from inference.flow_matching_sampler import FlowMatchingSampler
from dataset.transforms import denormalize
from utils.seed import set_seed


def create_fake_dataloader(num_samples: int = 32, batch_size: int = 8):
    """Create fake data for testing."""
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def test_ddpm(device: torch.device, output_dir: Path):
    """Test DDPM training and sampling."""
    print("\n" + "=" * 50)
    print("Testing DDPM")
    print("=" * 50)
    
    # Create model and trainer
    model = UNetSmall(num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    schedule = NoiseSchedule(num_timesteps=10)  # Very small for testing
    
    checkpoint_dir = output_dir / "ddpm"
    trainer = DDPMTrainer(
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    # Train for 1 epoch on fake data
    dataloader = create_fake_dataloader(num_samples=16, batch_size=4)
    
    print("Training for 1 epoch...")
    history = trainer.train(dataloader, num_epochs=1, save_every=1)
    print(f"  Final loss: {history['losses'][-1]:.4f}")
    
    # Verify checkpoint saved
    checkpoint_path = checkpoint_dir / "final.pt"
    assert checkpoint_path.exists(), "Checkpoint not saved!"
    print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Load and sample
    print("Loading checkpoint and sampling...")
    model2 = UNetSmall(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model2.load_state_dict(checkpoint["model_state_dict"])
    
    sampler = DDPMSampler(model2, schedule, device)
    samples = sampler.sample(
        batch_size=4,
        class_label=torch.tensor([0, 1, 2, 3]),
        show_progress=False,
    )
    
    assert samples.shape == (4, 3, 32, 32), f"Wrong shape: {samples.shape}"
    print(f"  Generated {samples.shape[0]} samples ✓")
    
    print("DDPM test passed! ✓")
    return True


def test_flow_matching(device: torch.device, output_dir: Path):
    """Test Flow Matching training and sampling."""
    print("\n" + "=" * 50)
    print("Testing Flow Matching")
    print("=" * 50)
    
    # Create model and trainer
    model = UNetSmall(num_classes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    checkpoint_dir = output_dir / "flow_matching"
    trainer = FlowMatchingTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
    )
    
    # Train for 1 epoch
    dataloader = create_fake_dataloader(num_samples=16, batch_size=4)
    
    print("Training for 1 epoch...")
    history = trainer.train(dataloader, num_epochs=1, save_every=1)
    print(f"  Final loss: {history['losses'][-1]:.4f}")
    
    # Verify checkpoint
    checkpoint_path = checkpoint_dir / "final.pt"
    assert checkpoint_path.exists(), "Checkpoint not saved!"
    print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Load and sample
    print("Loading checkpoint and sampling...")
    model2 = UNetSmall(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model2.load_state_dict(checkpoint["model_state_dict"])
    
    sampler = FlowMatchingSampler(model2, device)
    samples = sampler.sample(
        batch_size=4,
        num_steps=5,  # Very few steps for speed
        class_label=torch.tensor([0, 1, 2, 3]),
        show_progress=False,
    )
    
    assert samples.shape == (4, 3, 32, 32), f"Wrong shape: {samples.shape}"
    print(f"  Generated {samples.shape[0]} samples ✓")
    
    print("Flow Matching test passed! ✓")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")
    
    set_seed(42)
    
    # Use temp directory for test outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        try:
            test_ddpm(device, output_dir)
            test_flow_matching(device, output_dir)
            
            print("\n" + "=" * 50)
            print("ALL TESTS PASSED! ✓")
            print("=" * 50)
            print("\nThe pipeline is ready for training.")
            print("Run on Colab with GPU:")
            print("  python scripts/train.py --method ddpm --device cuda")
            print("  python scripts/train.py --method flow_matching --device cuda")
            
        except Exception as e:
            print(f"\nTEST FAILED: {e}")
            raise


if __name__ == "__main__":
    main()
