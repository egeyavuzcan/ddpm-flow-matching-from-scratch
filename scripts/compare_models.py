#!/usr/bin/env python
"""
Compare DDPM and Flow Matching models.

Generates:
1. Side-by-side sample comparison for each class
2. Loss curve comparison (from TensorBoard logs)
3. Timing comparison

Usage:
    python scripts/compare_models.py
"""
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling.unet.unet import UNetSmall
from modeling.model_factory import create_model
from diffusion.ddpm.noise_schedule import NoiseSchedule
from inference.ddpm_sampler import DDPMSampler
from inference.flow_matching_sampler import FlowMatchingSampler
from dataset.transforms import denormalize
from utils.seed import set_seed


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    model = create_model("unet_small", num_classes=10, image_size=32)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint.get("epoch", "unknown")


def generate_samples(
    sampler,
    classes: list,
    num_per_class: int = 4,
    **kwargs
) -> torch.Tensor:
    """Generate samples for specified classes."""
    all_samples = []
    
    for class_idx in classes:
        class_labels = torch.tensor([class_idx] * num_per_class)
        samples = sampler.sample(
            batch_size=num_per_class,
            class_label=class_labels,
            show_progress=False,
            **kwargs
        )
        all_samples.append(samples)
    
    return torch.cat(all_samples, dim=0)


def samples_to_grid(samples: torch.Tensor, nrow: int = 4) -> np.ndarray:
    """Convert samples to grid image."""
    samples = denormalize(samples).cpu().numpy()
    samples = (samples * 255).astype(np.uint8)
    samples = samples.transpose(0, 2, 3, 1)  # BCHW -> BHWC
    
    n = len(samples)
    h, w, c = samples[0].shape
    ncol = nrow
    nrows = (n + ncol - 1) // ncol
    
    grid = np.zeros((nrows * h, ncol * w, c), dtype=np.uint8)
    
    for i, sample in enumerate(samples):
        row = i // ncol
        col = i % ncol
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = sample
    
    return grid


def create_comparison_grid(
    ddpm_samples: torch.Tensor,
    fm_samples: torch.Tensor,
    classes: list,
    num_per_class: int = 4,
) -> np.ndarray:
    """Create side-by-side comparison grid."""
    # Denormalize
    ddpm = denormalize(ddpm_samples).cpu().numpy()
    fm = denormalize(fm_samples).cpu().numpy()
    
    ddpm = (ddpm * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    fm = (fm * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    
    h, w, c = ddpm[0].shape
    num_classes = len(classes)
    
    # Grid: rows = classes, cols = DDPM samples + separator + FM samples
    # Add labels
    label_height = 20
    sep_width = 10
    
    grid_width = (num_per_class * 2 + 1) * w + sep_width
    grid_height = num_classes * h + label_height
    
    grid = np.ones((grid_height, grid_width, c), dtype=np.uint8) * 255
    
    # Add samples
    for class_i, class_idx in enumerate(classes):
        row_y = label_height + class_i * h
        
        # DDPM samples
        for j in range(num_per_class):
            idx = class_i * num_per_class + j
            col_x = j * w
            grid[row_y:row_y+h, col_x:col_x+w] = ddpm[idx]
        
        # Separator
        sep_x = num_per_class * w
        grid[row_y:row_y+h, sep_x:sep_x+sep_width] = 128
        
        # FM samples
        for j in range(num_per_class):
            idx = class_i * num_per_class + j
            col_x = sep_x + sep_width + j * w
            grid[row_y:row_y+h, col_x:col_x+w] = fm[idx]
    
    return grid


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm_checkpoint", type=str, 
                       default="outputs/ddpm/checkpoints/final.pt")
    parser.add_argument("--fm_checkpoint", type=str,
                       default="outputs/flow/checkpoints/final (1).pt")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_per_class", type=int, default=4)
    parser.add_argument("--ddpm_steps", type=int, default=1000)
    parser.add_argument("--fm_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\n=== Loading Models ===")
    print(f"DDPM checkpoint: {args.ddpm_checkpoint}")
    ddpm_model, ddpm_epoch = load_model(args.ddpm_checkpoint, device)
    print(f"  Loaded (epoch {ddpm_epoch})")
    
    print(f"Flow Matching checkpoint: {args.fm_checkpoint}")
    fm_model, fm_epoch = load_model(args.fm_checkpoint, device)
    print(f"  Loaded (epoch {fm_epoch})")
    
    # Create samplers
    schedule = NoiseSchedule(num_timesteps=args.ddpm_steps, schedule_type="cosine")
    ddpm_sampler = DDPMSampler(ddpm_model, schedule.to(device), device)
    fm_sampler = FlowMatchingSampler(fm_model, device)
    
    # Generate samples for all classes
    print("\n=== Generating Samples ===")
    classes = list(range(10))  # All CIFAR-10 classes
    
    # DDPM sampling
    print(f"DDPM ({args.ddpm_steps} steps)...")
    ddpm_start = time.time()
    ddpm_samples = generate_samples(
        ddpm_sampler, classes, args.num_per_class,
        image_size=32
    )
    ddpm_time = time.time() - ddpm_start
    print(f"  Time: {ddpm_time:.1f}s ({ddpm_time/len(ddpm_samples):.2f}s per sample)")
    
    # Flow Matching sampling
    print(f"Flow Matching ({args.fm_steps} steps)...")
    fm_start = time.time()
    fm_samples = generate_samples(
        fm_sampler, classes, args.num_per_class,
        image_size=32, num_steps=args.fm_steps
    )
    fm_time = time.time() - fm_start
    print(f"  Time: {fm_time:.1f}s ({fm_time/len(fm_samples):.2f}s per sample)")
    print(f"  Speedup: {ddpm_time/fm_time:.1f}x faster!")
    
    # Save individual grids
    print("\n=== Saving Results ===")
    
    ddpm_grid = samples_to_grid(ddpm_samples, nrow=args.num_per_class)
    fm_grid = samples_to_grid(fm_samples, nrow=args.num_per_class)
    
    Image.fromarray(ddpm_grid).save(output_dir / "ddpm_samples.png")
    Image.fromarray(fm_grid).save(output_dir / "flow_matching_samples.png")
    print(f"  Saved: {output_dir / 'ddpm_samples.png'}")
    print(f"  Saved: {output_dir / 'flow_matching_samples.png'}")
    
    # Create comparison for each class
    for class_idx in classes:
        class_name = CIFAR10_CLASSES[class_idx]
        
        # Get samples for this class
        start_idx = class_idx * args.num_per_class
        end_idx = start_idx + args.num_per_class
        
        ddpm_class = ddpm_samples[start_idx:end_idx]
        fm_class = fm_samples[start_idx:end_idx]
        
        # Combine horizontally
        ddpm_row = samples_to_grid(ddpm_class, nrow=args.num_per_class)
        fm_row = samples_to_grid(fm_class, nrow=args.num_per_class)
        
        # Stack vertically with labels
        h1, w1 = ddpm_row.shape[:2]
        h2, w2 = fm_row.shape[:2]
        
        combined = np.ones((h1 + h2 + 10, max(w1, w2), 3), dtype=np.uint8) * 255
        combined[:h1, :w1] = ddpm_row
        combined[h1+10:h1+10+h2, :w2] = fm_row
        
        Image.fromarray(combined).save(output_dir / f"class_{class_idx}_{class_name}.png")
    
    print(f"  Saved class comparisons to {output_dir}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"DDPM: {args.ddpm_steps} steps, {ddpm_time:.1f}s total")
    print(f"Flow Matching: {args.fm_steps} steps, {fm_time:.1f}s total")
    print(f"FM is {ddpm_time/fm_time:.1f}x faster than DDPM")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
