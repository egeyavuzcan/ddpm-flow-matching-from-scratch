import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import numpy as np

def extract_scalars(log_file, tag='train/loss'):
    ea = EventAccumulator(log_file)
    ea.Reload()
    
    tags = ea.Tags()['scalars']
    print(f"Available tags in {os.path.basename(log_file)}: {tags}")
    
    if tag in tags:
        return [(s.step, s.value) for s in ea.Scalars(tag)]
    
    # Try finding similar tags
    for t in tags:
        if 'loss' in t.lower():
            print(f"Found similar tag: {t}")
            return [(s.step, s.value) for s in ea.Scalars(t)]
            
    return []

def main():
    flow_log = r"c:\Users\egeya\Code Base\ddpm-flow-matching-from-scratch\outputs\dit_models\flow\events.out.tfevents.1767514400.98a1860a32a8.3860.0"
    ddpm_log = r"c:\Users\egeya\Code Base\ddpm-flow-matching-from-scratch\outputs\dit_models\ddpm\events.out.tfevents.1767522583.98a1860a32a8.41114.0"
    
    print("Extracting Flow Matching logs...")
    flow_data = extract_scalars(flow_log)
    print(f"Flow Matching: {len(flow_data)} points")
    
    print("Extracting DDPM logs...")
    ddpm_data = extract_scalars(ddpm_log)
    print(f"DDPM: {len(ddpm_data)} points")
    
    if not flow_data and not ddpm_data:
        print("Error: Could not extract data from either log")
        return

    # Create plot
    plt.figure(figsize=(10, 6))
    
    if flow_data:
        steps_f, values_f = zip(*flow_data)
        plt.plot(steps_f, values_f, 'b-', alpha=0.3)
        plt.plot(steps_f, smooth(values_f, 50), 'b-', label='Flow Matching (DiT-S)')
        print("\nFlow Matching (DiT-S) Stats:")
        print(f"Initial Loss: {values_f[0]:.4f}")
        print(f"Final Loss: {values_f[-1]:.4f}")
    else:
        print("No Flow Matching data found.")
    
    if ddpm_data:
        steps_d, values_d = zip(*ddpm_data)
        plt.plot(steps_d, values_d, 'r-', alpha=0.3)
        plt.plot(steps_d, smooth(values_d, 50), 'r-', label='DDPM (DiT-S)')
        print("\nDDPM (DiT-S) Stats:")
        print(f"Initial Loss: {values_d[0]:.4f}")
        print(f"Final Loss: {values_d[-1]:.4f}")
    else:
        print("No DDPM data found.")
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss: DiT-S (DDPM vs Flow Matching)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('outputs/dit_comparison', exist_ok=True)
    plt.savefig('outputs/dit_comparison/dit_training_loss.png')
    print("Saved plot to outputs/dit_comparison/dit_training_loss.png")
    


if __name__ == "__main__":
    main()
