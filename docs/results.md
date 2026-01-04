# DDPM vs Flow Matching: Training Results

This report contains a comparison of **DDPM** and **Flow Matching** methods under the same UNet model and training conditions.

---

## 🔬 Experiment Setup

| Parameter | Value |
|-----------|-------|
| Model | UNetSmall (~2.7M parameters) |
| Dataset | CIFAR-10 (50K images) |
| Image Size | 32×32 |
| Epochs | 100 |
| Batch Size | 128 |
| Learning Rate | 0.0002 |
| Optimizer | AdamW |

---

## 📊 Training Loss Comparison

### DDPM
```
Initial Loss:  0.1299
Final Loss:    0.0333
Min Loss:      0.0178 (step 22700)
Improvement:   74% ↓
Trend:         Still decreasing (slope: -0.000005)
```

### Flow Matching
```
Initial Loss:  0.4292
Final Loss:    0.1606
Min Loss:      0.1567 (step 38600)
Improvement:   62% ↓
Trend:         Still decreasing (slope: -0.000054)
```

### Analysis

| Metric | DDPM | Flow Matching | Comment |
|--------|------|---------------|-------|
| Final Loss | 0.033 | 0.161 | DDPM loss is lower |
| Visual Quality | Noisy | Smooth/Coherent | **FM is much better!** |
| Trend | Decreasing slowly | Decreasing fast | FM learns faster |

---

## 🤔 Why Does Flow Matching Give Better Results?

### 1. Loss Values Are Misleading

**DDPM loss is lower but visual quality is worse. Why?**

- **DDPM:** Predicts noise (`ε_θ`)
- **Flow Matching:** Predicts velocity (`v_θ`)

These two values are on different scales:
- Noise: `~N(0, 1)` - usually small values
- Velocity: `x_1 - x_0` - larger range

**Conclusion:** Comparing loss values directly is meaningless!

### 2. Difference in Sampling Steps

Steps used during testing:

| Method | Steps | Per-sample Time |
|--------|-------|-----------------|
| DDPM | 100 | 0.88s |
| Flow Matching | 20 | 0.16s |

**DDPM was trained with 1000 steps but tested with 100 steps!**

This causes significant quality loss because:
- DDPM is optimized for a 1000-step Markov chain
- Running with 100 steps skips intermediate steps
- The model cannot compensate for this "skipping"

### 3. Natural Advantage of Flow Matching

```
DDPM:          Discrete steps, Markov chain
Flow Matching: Continuous ODE, smooth trajectory
```

**Flow Matching advantages:**

1. **Linear Path:** `x_t = (1-t)·x_0 + t·x_1`
   - A straight line, easy to learn
   - Velocity is constant everywhere

2. **Flexible Sampling:**
   - Works with any number of steps
   - Even 20 steps give good results

3. **Smoother Trajectories:**
   - ODE solution is more stable
   - Even Euler method is sufficient

### 4. Why Did DDPM Fail?

```
Training:  t ∈ {0, 1, 2, ..., 999} (1000 discrete steps)
Testing:   t ∈ {0, 10, 20, ..., 990} (100 steps, skipping 10)
```

The DDPM model learned to predict noise at `t=500`.
But during testing, `t=500` is skipped, and the model sees `t=490` and `t=510`.

**Solution:** For DDPM, either:
- Use 1000 steps (very slow)
- Use DDIM sampler (adaptive)
- Train with fewer timesteps

---

## 📈 Result Images

### Flow Matching Samples (Good Quality)
![Flow Matching Samples](outputs/comparison/flow_matching_samples.png)

### DDPM Samples (Noisy)
![DDPM Samples](outputs/comparison/ddpm_samples.png)

### Class-Based Comparison

Top row DDPM, bottom row Flow Matching for each class:

| Class | Comparison |
|-------|---------------|
| Airplane | ![](outputs/comparison/class_0_airplane.png) |
| Automobile | ![](outputs/comparison/class_1_automobile.png) |
| Cat | ![](outputs/comparison/class_3_cat.png) |
| Dog | ![](outputs/comparison/class_5_dog.png) |

---

## ⏱️ Speed Comparison

| Method | Steps | Total Time | Per Sample |
|-------|------|-------------|--------------|
| DDPM (100 steps) | 100 | 17.6s | 0.88s |
| Flow Matching (20 steps) | 20 | 3.3s | 0.16s |
| **Speedup** | **5x** | **5.4x** | **5.5x** |

---

## 🎯 Recommendations

### To Improve DDPM:
1. **Use 1000 steps:** `--ddpm_steps 1000` (slow but correct)
2. **Add DDIM Sampler:** Gives good results with fewer steps
3. **Use Cosine Schedule:** Smoother transitions

### To Improve Flow Matching:
1. **More epochs:** Try 200-300 epochs
2. **Heun solver:** `--solver heun` (2x slower, better quality)
3. **Larger model:** Use `unet` instead of `unet_small`

### General:
- **EMA (Exponential Moving Average):** More stable results
- **CFG (Classifier-Free Guidance):** Sharper images
- **Learning Rate Scheduling:** Cosine decay

---

## 🆕 DiT (Diffusion Transformer) Results

We also trained a **DiT-S** (Small) model for 200 epochs to compare with UNet.

### Training Loss Comparison (DiT-S)
![DiT Loss](outputs/dit_comparison/dit_training_loss.png)

| Model | Method | Epochs | Final Loss | Comment |
|-------|--------|--------|------------|---------|
| **UNet (Small)** | DDPM | 100 | **0.033** | 🏆 Best convergence |
| **DiT-S** | DDPM | 200 | 0.054 | Slower convergence (Transformers need more training) |
| **UNet (Small)** | Flow Matching | 100 | 0.161 | Loss scale not comparable to DDPM |

**Key Observations:**
1.  **Convergence:** DiT-S has a higher final loss (0.054) than UNet (0.033) even with double the epochs (200 vs 100). This is expected as Transformers typically lack the inductive bias of CNNs and require more training to learn spatial relationships.
2.  **Architecture:** DiT uses `patch_size=4` and processes images as sequences, whereas UNet processes them as 2D maps.

### Visual Comparison (Frog)
![DiT Comparison Frog](outputs/dit_comparison/class_6_frog.png)
*Left: DDPM (Noisy), Right: Flow Matching (Clean)*

---

## 📝 Conclusion

| Criterion | Winner | Reason |
|--------|---------|-------|
| **Visual Quality** | 🏆 Flow Matching | Smooth, recognizable images |
| **Training Speed** | Tie | Same number of epochs |
| **Sampling Speed** | 🏆 Flow Matching | 5.4x faster |
| **Flexibility** | 🏆 Flow Matching | Works with any step count |
| **Theoretical Beauty** | DDPM | Deep mathematical foundations |

**Conclusion:** **Flow Matching** should be preferred for practical applications.

---

## 📚 References

1. [DDPM - Ho et al., 2020](https://arxiv.org/abs/2006.11239)
2. [Flow Matching - Lipman et al., 2022](https://arxiv.org/abs/2210.02747)
3. [DDIM - Song et al., 2020](https://arxiv.org/abs/2010.02502)
