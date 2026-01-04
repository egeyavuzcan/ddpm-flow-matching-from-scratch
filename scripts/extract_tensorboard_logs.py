#!/usr/bin/env python
"""
Extract loss data from TensorBoard logs and create comparison plots.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def extract_scalar_data(log_file):
    """Extract scalar data from TensorBoard log file."""
    ea = event_accumulator.EventAccumulator(str(log_file))
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()['scalars']
    
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data

def plot_training_comparison(ddpm_data, flow_data, output_path):
    """Create comparison plot of training losses."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # DDPM plot
    if 'Loss/train' in ddpm_data:
        steps = ddpm_data['Loss/train']['steps']
        values = ddpm_data['Loss/train']['values']
        ax1.plot(steps, values, alpha=0.6, linewidth=0.5, color='#1f77b4')
        # Smooth curve
        window = min(100, len(values) // 10)
        if window > 1:
            smooth = np.convolve(values, np.ones(window)/window, mode='valid')
            ax1.plot(steps[:len(smooth)], smooth, linewidth=2, color='#1f77b4', label='DDPM')
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('DDPM Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Flow Matching plot
    if 'Loss/train' in flow_data:
        steps = flow_data['Loss/train']['steps']
        values = flow_data['Loss/train']['values']
        ax2.plot(steps, values, alpha=0.6, linewidth=0.5, color='#ff7f0e')
        # Smooth curve
        window = min(100, len(values) // 10)
        if window > 1:
            smooth = np.convolve(values, np.ones(window)/window, mode='valid')
            ax2.plot(steps[:len(smooth)], smooth, linewidth=2, color='#ff7f0e', label='Flow Matching')
    
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.set_title('Flow Matching Training Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {output_path}")
    plt.close()

def plot_combined_comparison(ddpm_data, flow_data, output_path):
    """Create single plot comparing both methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # DDPM
    if 'Loss/train' in ddpm_data:
        steps = ddpm_data['Loss/train']['steps']
        values = ddpm_data['Loss/train']['values']
        # Convert steps to epochs (assuming ~390 steps per epoch for CIFAR-10)
        epochs = np.array(steps) / 390
        window = min(100, len(values) // 10)
        if window > 1:
            smooth = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(epochs[:len(smooth)], smooth, linewidth=2.5, color='#1f77b4', 
                   label='DDPM (Noise Prediction)', alpha=0.9)
    
    # Flow Matching
    if 'Loss/train' in flow_data:
        steps = flow_data['Loss/train']['steps']
        values = flow_data['Loss/train']['values']
        epochs = np.array(steps) / 390
        window = min(100, len(values) // 10)
        if window > 1:
            smooth = np.convolve(values, np.ones(window)/window, mode='valid')
            ax.plot(epochs[:len(smooth)], smooth, linewidth=2.5, color='#2ca02c', 
                   label='Flow Matching (Velocity Prediction)', alpha=0.9)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Training Loss (MSE)', fontsize=13, fontweight='bold')
    ax.set_title('DDPM vs Flow Matching: Training Loss Comparison', 
                fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")
    plt.close()

def main():
    # Paths
    project_root = Path(__file__).parent.parent
    ddpm_log = project_root / "outputs/ddpm/logs/events.out.tfevents.1767473814.862102c20a9f.5501.0"
    flow_log = project_root / "outputs/flow/logs/events.out.tfevents.1767475244.862102c20a9f.13446.0"
    output_dir = project_root / "outputs/results"
    output_dir.mkdir(exist_ok=True)
    
    # Extract data
    print("Extracting DDPM logs...")
    ddpm_data = extract_scalar_data(ddpm_log)
    
    print("Extracting Flow Matching logs...")
    flow_data = extract_scalar_data(flow_log)
    
    # Create plots
    print("\nGenerating plots...")
    plot_training_comparison(ddpm_data, flow_data, output_dir / "training_loss_comparison.png")
    plot_combined_comparison(ddpm_data, flow_data, output_dir / "combined_loss_plot.png")
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    
    if 'Loss/train' in ddpm_data:
        values = ddpm_data['Loss/train']['values']
        print(f"\nDDPM:")
        print(f"  Initial Loss: {values[0]:.4f}")
        print(f"  Final Loss:   {values[-1]:.4f}")
        print(f"  Min Loss:     {min(values):.4f}")
        print(f"  Total Steps:  {len(values)}")
    
    if 'Loss/train' in flow_data:
        values = flow_data['Loss/train']['values']
        print(f"\nFlow Matching:")
        print(f"  Initial Loss: {values[0]:.4f}")
        print(f"  Final Loss:   {values[-1]:.4f}")
        print(f"  Min Loss:     {min(values):.4f}")
        print(f"  Total Steps:  {len(values)}")
    
    print("\n" + "="*60)
    print(f"Plots saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
