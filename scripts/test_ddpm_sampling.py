#!/usr/bin/env python
"""
Test DDPM sampling with different step counts to diagnose noisy outputs.
"""
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling.model_factory import create_model
from diffusion.ddpm.noise_schedule import NoiseSchedule
from dataset.transforms import denormalize


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
    print(f"✓ Saved: {path}")


@torch.no_grad()
def ddpm_sample_with_stride(model, schedule, device, batch_size=16, num_steps=100, class_labels=None):
    """
    DDPM sampling with step striding (CORRECT implementation).
    
    Key: Use the SAME schedule as training (1000 timesteps), but skip steps.
    """
    model.eval()
    
    # Start from pure noise
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    
    if class_labels is not None:
        class_labels = class_labels.to(device)
    
    # Calculate stride to get desired number of steps
    total_timesteps = schedule.num_timesteps
    stride = total_timesteps // num_steps
    
    # Create timestep sequence (e.g., [999, 989, 979, ..., 9])
    timesteps = list(range(total_timesteps - 1, 0, -stride))
    
    print(f"Sampling with {len(timesteps)} steps (stride={stride})")
    print(f"Timestep sequence: {timesteps[:5]} ... {timesteps[-5:]}")
    
    for t in tqdm(timesteps, desc=f"DDPM ({num_steps} steps)"):
        # Create timestep tensor
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = model(x, t_tensor, class_labels)
        
        # Get coefficients from schedule
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bar[t]
        beta = schedule.betas[t]
        
        # Compute mean: μ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε_θ)
        sqrt_recip_alpha = 1.0 / torch.sqrt(alpha)
        noise_coef = beta / torch.sqrt(1.0 - alpha_bar)
        mean = sqrt_recip_alpha * (x - noise_coef * noise_pred)
        
        # Add noise (except at final step)
        if t > 1:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta)
            x = mean + sigma * noise
        else:
            x = mean
    
    return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Load model
    checkpoint_path = Path("outputs/ddpm/checkpoints/final.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    model = create_model(model_type="unet_small", num_classes=10, image_size=32)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}\n")
    
    # Create schedule (SAME as training: 1000 timesteps, cosine)
    schedule = NoiseSchedule(num_timesteps=1000, schedule_type="cosine")
    schedule = schedule.to(device)
    
    # Test different step counts
    step_counts = [1000, 500, 250, 100, 50]
    
    # Generate samples for each class
    class_labels = torch.arange(0, 10, device=device).repeat(2)[:16]  # 16 samples
    
    output_dir = Path("outputs/ddpm_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Testing DDPM with different step counts")
    print("="*60)
    
    for num_steps in step_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {num_steps} steps")
        print(f"{'='*60}")
        
        samples = ddpm_sample_with_stride(
            model=model,
            schedule=schedule,
            device=device,
            batch_size=16,
            num_steps=num_steps,
            class_labels=class_labels
        )
        
        # Denormalize and save
        samples = denormalize(samples)
        samples = samples.cpu().numpy()
        samples = (samples * 255).astype(np.uint8)
        samples = samples.transpose(0, 2, 3, 1)  # BCHW -> BHWC
        
        save_image_grid(samples, output_dir / f"ddpm_{num_steps}_steps.png", nrow=4)
    
    print("\n" + "="*60)
    print("✓ All tests complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
