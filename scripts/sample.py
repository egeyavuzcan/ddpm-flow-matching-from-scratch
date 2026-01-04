#!/usr/bin/env python
"""
Sampling script for DDPM and Flow Matching models.

Usage:
    python scripts/sample.py --method ddpm --checkpoint outputs/ddpm/checkpoints/final.pt
    python scripts/sample.py --method flow_matching --checkpoint outputs/flow_matching/checkpoints/final.pt
    
    # Generate specific classes
    python scripts/sample.py --method ddpm --checkpoint model.pt --classes 0 1 2 3 4
    
    # Quick test
    python scripts/sample.py --method ddpm --checkpoint model.pt --num_samples 4 --steps 10
"""
import argparse
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling.unet.unet import UNet, UNetSmall
from modeling.model_factory import create_model
from diffusion.ddpm.noise_schedule import NoiseSchedule
from inference.ddpm_sampler import DDPMSampler
from inference.flow_matching_sampler import FlowMatchingSampler
from dataset.transforms import denormalize
from utils.seed import set_seed


def save_samples(samples: torch.Tensor, output_dir: Path, prefix: str = "sample"):
    """Save generated samples as images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize from [-1, 1] to [0, 1]
    samples = denormalize(samples)
    
    # Convert to numpy
    samples = samples.cpu().numpy()
    samples = (samples * 255).astype(np.uint8)
    samples = samples.transpose(0, 2, 3, 1)  # BCHW -> BHWC
    
    # Save individual images
    paths = []
    for i, sample in enumerate(samples):
        path = output_dir / f"{prefix}_{i:04d}.png"
        Image.fromarray(sample).save(path)
        paths.append(path)
    
    # Save as grid
    grid_path = output_dir / f"{prefix}_grid.png"
    save_image_grid(samples, grid_path)
    
    return paths, grid_path


def save_image_grid(samples: np.ndarray, path: Path, nrow: int = 8):
    """Save samples as a grid image."""
    n = len(samples)
    ncol = min(nrow, n)
    nrow = (n + ncol - 1) // ncol
    
    h, w, c = samples[0].shape
    grid = np.zeros((nrow * h, ncol * w, c), dtype=np.uint8)
    
    for i, sample in enumerate(samples):
        row = i // ncol
        col = i % ncol
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = sample
    
    Image.fromarray(grid).save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained model")
    parser.add_argument("--method", type=str, choices=["ddpm", "flow_matching"],
                       required=True, help="Sampling method")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=64,
                       help="Number of samples to generate")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                       help="Class labels to generate (cycles through if fewer than num_samples)")
    parser.add_argument("--steps", type=int, default=None,
                       help="Number of sampling steps (DDPM: 1000, FM: 50)")
    parser.add_argument("--solver", type=str, choices=["euler", "heun"], default="euler",
                       help="ODE solver for Flow Matching")
    parser.add_argument("--output_dir", type=str, default="./outputs/samples",
                       help="Output directory for samples")
    parser.add_argument("--model_type", type=str, default="unet_small",
                       help="Model type (must match checkpoint)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = create_model(
        model_type=args.model_type,
        num_classes=10,  # CIFAR-10
        image_size=32,
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Prepare class labels
    if args.classes is not None:
        # Repeat classes to match num_samples
        classes = args.classes * (args.num_samples // len(args.classes) + 1)
        classes = classes[:args.num_samples]
        class_labels = torch.tensor(classes, device=device)
    else:
        # Random classes
        class_labels = torch.randint(0, 10, (args.num_samples,), device=device)
    
    print(f"Generating {args.num_samples} samples...")
    print(f"Classes: {class_labels[:10].tolist()}{'...' if len(class_labels) > 10 else ''}")
    
    # Create sampler and generate
    if args.method == "ddpm":
        inference_steps = args.steps or 1000
        print(f"Using DDPM sampler with {inference_steps} inference steps...")
        
        # IMPORTANT: Use the SAME schedule as training (1000 timesteps)
        # The sampler will skip steps if inference_steps < 1000
        schedule = NoiseSchedule(num_timesteps=1000, schedule_type="cosine")
        sampler = DDPMSampler(model, schedule, device)
        
        samples = sampler.sample(
            batch_size=args.num_samples,
            image_size=32,
            class_label=class_labels,
            show_progress=True,
            num_inference_steps=inference_steps,
        )
    else:  # flow_matching
        steps = args.steps or 50
        print(f"Using Flow Matching sampler with {steps} steps ({args.solver} solver)...")
        
        sampler = FlowMatchingSampler(model, device)
        
        samples = sampler.sample(
            batch_size=args.num_samples,
            image_size=32,
            num_steps=steps,
            class_label=class_labels,
            show_progress=True,
            solver=args.solver,
        )
    
    # Save samples
    output_dir = Path(args.output_dir) / args.method
    print(f"\nSaving samples to {output_dir}...")
    
    paths, grid_path = save_samples(samples, output_dir, prefix=f"sample_{args.method}")
    
    print(f"Saved {len(paths)} individual images")
    print(f"Saved grid: {grid_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
