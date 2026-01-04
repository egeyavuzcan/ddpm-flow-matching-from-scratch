#!/usr/bin/env python
"""
Analyze training loss curves from TensorBoard logs.
"""
import sys
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

def read_tensorboard_logs(log_dir: str):
    """Read loss values from TensorBoard events file."""
    log_path = Path(log_dir)
    event_files = list(log_path.glob("events.out.tfevents.*"))
    
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None, None
    
    ea = event_accumulator.EventAccumulator(str(log_path))
    ea.Reload()
    
    # Get available scalars
    scalar_tags = ea.Tags().get('scalars', [])
    print(f"Available scalars: {scalar_tags}")
    
    losses = []
    steps = []
    
    if 'Loss/train' in scalar_tags:
        events = ea.Scalars('Loss/train')
        for event in events:
            steps.append(event.step)
            losses.append(event.value)
    
    return np.array(steps), np.array(losses)


def analyze_loss_curve(steps, losses, name):
    """Analyze loss curve and provide recommendations."""
    print(f"\n{'='*50}")
    print(f"  {name} Loss Analysis")
    print(f"{'='*50}")
    
    if len(losses) == 0:
        print("No loss data found!")
        return
    
    # Basic stats
    print(f"  Total steps logged: {len(losses)}")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Min loss: {losses.min():.4f} (step {steps[losses.argmin()]})")
    print(f"  Max loss: {losses.max():.4f}")
    
    # Check convergence
    # Look at last 20% vs first 20%
    split = len(losses) // 5
    early_mean = losses[:split].mean()
    late_mean = losses[-split:].mean()
    improvement = (early_mean - late_mean) / early_mean * 100
    
    print(f"\n  Early avg (first 20%): {early_mean:.4f}")
    print(f"  Late avg (last 20%): {late_mean:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    # Check if still decreasing
    last_quarter = losses[-len(losses)//4:]
    if len(last_quarter) > 10:
        trend = np.polyfit(range(len(last_quarter)), last_quarter, 1)[0]
        is_decreasing = trend < 0
        
        print(f"\n  Last 25% trend: {'↓ decreasing' if is_decreasing else '→ plateaued'}")
        print(f"  Slope: {trend:.6f}")
    
    # Recommendations
    print(f"\n  Recommendation:")
    
    if late_mean < 0.1 and improvement > 50:
        print("    ✓ Training looks good! Model has converged well.")
        print("    ✓ No regularization needed yet.")
    elif improvement < 20:
        print("    ⚠ Limited improvement - model may need more capacity or data augmentation")
    elif trend < -0.0001 if 'trend' in dir() else False:
        print("    → Loss still decreasing - more training would help!")
        print("    → Try 200-300 epochs for better results")
    else:
        print("    → Model has converged")
        print("    → For better quality, try:")
        print("      - Larger model (unet instead of unet_small)")
        print("      - EMA (Exponential Moving Average)")
        print("      - Cosine learning rate schedule")


def main():
    ddpm_log_dir = "outputs/ddpm/logs"
    flow_log_dir = "outputs/flow/logs"
    
    print("Reading TensorBoard logs...")
    
    # DDPM
    if Path(ddpm_log_dir).exists():
        steps, losses = read_tensorboard_logs(ddpm_log_dir)
        if losses is not None and len(losses) > 0:
            analyze_loss_curve(steps, losses, "DDPM")
    
    # Flow Matching
    if Path(flow_log_dir).exists():
        steps, losses = read_tensorboard_logs(flow_log_dir)
        if losses is not None and len(losses) > 0:
            analyze_loss_curve(steps, losses, "Flow Matching")
    
    print("\n" + "="*50)
    print("  To see visual loss curves:")
    print("  tensorboard --logdir outputs")
    print("="*50)


if __name__ == "__main__":
    main()
