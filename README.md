# DDPM vs Flow Matching: From Scratch

Complete implementations of **DDPM** and **Flow Matching** for CIFAR-10 image generation.

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/egeyavuzcan/ddpm-flow-matching-from-scratch.git
cd ddpm-flow-matching-from-scratch

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/quick_test.py
```

### Training 

```bash
# DDPM Training 
python scripts/train.py --method ddpm --device cuda

# Flow Matching Training 
python scripts/train.py --method flow_matching --device cuda

# Quick test (1 epoch)
python scripts/train.py --method ddpm --epochs 1 --device cuda
```

### Generate Samples

```bash
# DDPM Sampling (1000 steps)
python scripts/sample.py --method ddpm --checkpoint outputs/ddpm/checkpoints/final.pt

# Flow Matching Sampling (50 steps - 20x faster!)
python scripts/sample.py --method flow_matching --checkpoint outputs/flow_matching/checkpoints/final.pt

# Generate specific classes (0=airplane, 1=automobile, 5=dog, etc.)
python scripts/sample.py --method ddpm --checkpoint model.pt --classes 0 1 2 3 4 5 6 7 8 9
```

### View Training Logs

```bash
tensorboard --logdir outputs/ddpm/logs
```

---

## 📊 Project Structure

```
ddpm-flow-matching-from-scratch/
├── configs/              # YAML configurations
│   ├── base.yaml
│   ├── ddpm_cifar10.yaml
│   └── flow_matching_cifar10.yaml
├── scripts/              # CLI scripts
│   ├── train.py          # Training script
│   ├── sample.py         # Sampling script
│   └── quick_test.py     # Pipeline validation
├── src/
│   ├── dataset/          # CIFAR-10 data loading
│   ├── modeling/         # UNet architecture
│   ├── diffusion/        # DDPM & Flow Matching
│   ├── training/         # Trainers with TensorBoard
│   ├── inference/        # Samplers
│   └── utils/            # Utilities
├── tests/                # Unit & integration tests
└── docs/                 # Documentation
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [UNet Architecture](docs/unet_architecture.md) | Detailed UNet diagram with block explanations |
| [DDPM vs Flow Matching Formulas](docs/ddpm_flow_matching_formulas.md) | Mathematical reference for both methods |
| [Deep Dive (2x2 Examples)](docs/ddpm_vs_flow_matching_deep_dive.md) | Step-by-step computations |

---

## 🔧 Configuration

Key parameters in `configs/base.yaml`:

```yaml
model:
  type: "unet_small"     # ~2.7M params (use "unet" for full model)
  
training:
  epochs: 100
  learning_rate: 0.0002
  batch_size: 128

ddpm:
  num_timesteps: 1000
  schedule_type: "cosine"  # Better than linear

sampling:
  ddpm_steps: 1000        # DDPM: slow but stable
  fm_steps: 50            # Flow Matching: 20x faster!
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Quick pipeline test
python scripts/quick_test.py
```

---

## CIFAR-10 Classes

| Index | Class |
|-------|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

---

## References

- [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Flow Matching (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672)

---

## License

MIT
