# DDPM vs Flow Matching: Mathematical Formulas

This document provides a comprehensive mathematical reference for both diffusion methods implemented in this project.

---

## Table of Contents

1. [DDPM (Denoising Diffusion Probabilistic Models)](#ddpm)
2. [Flow Matching](#flow-matching)
3. [Training Comparison](#training-comparison)
4. [Inference Comparison](#inference-comparison)

---

## DDPM

### Core Idea

DDPM gradually adds Gaussian noise to data over T timesteps (forward process), then learns to reverse this process (reverse process).

### Forward Process (Adding Noise)

The forward process is a Markov chain that gradually adds noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

**Closed-form solution** (jump directly to any timestep t):

```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
```

Which gives us the **reparameterization trick**:

```
x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

where:
- ε ~ N(0, I)           # Standard Gaussian noise
- α_t = 1 - β_t         # Alpha from beta
- ᾱ_t = ∏_{s=1}^t α_s   # Cumulative product (alpha_bar)
```

### Noise Schedule

**Linear Schedule** (original DDPM):
```
β_t = β_start + (β_end - β_start) · t/T

Typical values:
- β_start = 0.0001
- β_end = 0.02
- T = 1000
```

**Cosine Schedule** (improved DDPM):
```
ᾱ_t = cos²((t/T + s) / (1+s) · π/2)

where s = 0.008 (small offset to prevent β_t = 0)
```

### Alpha Values at Key Timesteps

| t | ᾱ_t (linear) | √ᾱ_t | √(1-ᾱ_t) | Meaning |
|---|--------------|------|----------|---------|
| 0 | ~1.0 | ~1.0 | ~0.0 | Pure data |
| 250 | ~0.7 | ~0.84 | ~0.55 | Mostly data |
| 500 | ~0.3 | ~0.55 | ~0.84 | Mixed |
| 750 | ~0.1 | ~0.32 | ~0.95 | Mostly noise |
| 999 | ~0.02 | ~0.14 | ~0.99 | Almost pure noise |

### Reverse Process (Denoising)

The reverse process learns to denoise:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)
```

**Predicted mean**:
```
μ_θ(x_t, t) = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t))
```

**Sampling step**:
```
x_{t-1} = μ_θ(x_t, t) + σ_t · z

where:
- z ~ N(0, I) if t > 1, else z = 0
- σ_t = √β_t (simplified variance)
```

### Training Objective

**Simple loss** (predict noise):
```
L = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t)||² ]

where:
- t ~ Uniform(1, T)
- x_0 ~ q(x_0)        # Sample from data
- ε ~ N(0, I)         # Sample noise
- x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
```

---

## Flow Matching

### Core Idea

Flow Matching learns a velocity field that transports samples from noise distribution to data distribution along a probability path.

### Probability Path (Linear Interpolation)

```
x_t = (1 - t) · x_0 + t · x_1

where:
- x_0 = data sample
- x_1 = noise sample ~ N(0, I)
- t ∈ [0, 1]
```

**Key insight**: At t=0, we have pure data. At t=1, we have pure noise.

> **Note**: Flow Matching convention is opposite to DDPM!
> - DDPM: t=0 is data, t=T is noise
> - Flow Matching: t=0 is data, t=1 is noise

### Target Velocity Field

The conditional velocity field is simply:

```
u_t(x_t | x_0, x_1) = x_1 - x_0
```

**Key insight**: The velocity is **constant** (independent of t)! This is because we're using linear interpolation.

### Training Objective

**Conditional Flow Matching loss**:
```
L = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - (x_1 - x_0)||² ]

where:
- t ~ Uniform(0, 1)
- x_0 ~ q(x_0)        # Sample from data
- x_1 ~ N(0, I)       # Sample noise
- x_t = (1-t)·x_0 + t·x_1
```

### Inference (ODE Solving)

Generate samples by solving the ODE backwards:

```
dx/dt = v_θ(x, t)
```

**Euler method** (simple):
```
x_{t-Δt} = x_t - v_θ(x_t, t) · Δt

Starting from x_1 ~ N(0, I), solve to t=0
```

**Typical settings**:
- 20-50 steps (much fewer than DDPM's 1000!)
- Δt = 1/num_steps

---

## Training Comparison

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| **Timestep range** | t ∈ {0, 1, ..., T-1} discrete | t ∈ [0, 1] continuous |
| **Noisy sample** | x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε | x_t = (1-t)·x_0 + t·x_1 |
| **Model predicts** | Noise ε | Velocity v |
| **Target** | ε (sampled noise) | x_1 - x_0 (velocity) |
| **Loss** | MSE(ε, ε_θ) | MSE(v, v_θ) |

### Training Loop Pseudocode

**DDPM**:
```python
for x_0 in dataloader:
    t = randint(0, T, (B,))
    ε = randn_like(x_0)
    
    # Forward process
    x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1 - alpha_bar[t]) * ε
    
    # Predict noise
    ε_pred = model(x_t, t)
    
    # Loss
    loss = MSE(ε, ε_pred)
```

**Flow Matching**:
```python
for x_0 in dataloader:
    t = rand(B,)              # Continuous [0, 1]
    x_1 = randn_like(x_0)     # Noise
    
    # Interpolation
    x_t = (1 - t) * x_0 + t * x_1
    
    # Target velocity
    target = x_1 - x_0
    
    # Predict velocity
    v_pred = model(x_t, t)
    
    # Loss
    loss = MSE(target, v_pred)
```

---

## Inference Comparison

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| **Steps** | ~1000 | ~20-50 |
| **Method** | Iterative denoising | ODE solving |
| **Formula** | Complex (mean + variance) | Simple: x -= v·Δt |
| **Speed** | Slow | Fast (20-50x) |

### Inference Loop Pseudocode

**DDPM** (1000 steps):
```python
x = randn(B, 3, 32, 32)  # Start from noise

for t in reversed(range(T)):
    ε_pred = model(x, t)
    
    # Compute mean
    μ = (1/sqrt(α[t])) * (x - (β[t]/sqrt(1-ᾱ[t])) * ε_pred)
    
    # Add noise (except at t=0)
    if t > 0:
        x = μ + sqrt(β[t]) * randn_like(x)
    else:
        x = μ

return x
```

**Flow Matching** (20 steps):
```python
x = randn(B, 3, 32, 32)  # Start from noise
dt = 1.0 / num_steps

for i in range(num_steps):
    t = 1.0 - i * dt
    v_pred = model(x, t)
    x = x - v_pred * dt   # Euler step

return x
```

---

## Key Equations Summary

### DDPM

```
Forward:    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

Reverse:    x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ) + σ_t·z

Loss:       L = ||ε - ε_θ(x_t, t)||²
```

### Flow Matching

```
Interpolation:  x_t = (1-t) · x_0 + t · x_1

Velocity:       u = x_1 - x_0

ODE:            x_{t-Δt} = x_t - v_θ(x_t, t) · Δt

Loss:           L = ||v_θ(x_t, t) - (x_1 - x_0)||²
```

---

## Implementation Notes

1. **Shared Model Architecture**: Both methods use the same UNet! Only the loss and sampling differ.

2. **Time Embedding**: 
   - DDPM: Integer timesteps [0, 999]
   - FM: Float timesteps [0.0, 1.0]
   - Model handles both via sinusoidal embedding

3. **Why Flow Matching is Faster**:
   - Linear path → smooth velocity field
   - Easier for model to learn
   - Requires fewer discretization steps

4. **Quality Trade-off**:
   - Both achieve similar quality
   - FM gets there 20-50x faster
   - DDPM can be accelerated with DDIM

---

## References

- **DDPM**: Ho et al., 2020 - [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- **Flow Matching**: Lipman et al., 2022 - [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
- **Improved DDPM**: Nichol & Dhariwal, 2021 - [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
