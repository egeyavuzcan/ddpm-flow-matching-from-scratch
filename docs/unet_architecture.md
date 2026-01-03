# UNet Architecture Deep Dive

Bu dokümanda, diffusion modelleri için implement ettiğimiz UNet mimarisinin detaylı açıklamasını bulacaksınız.

---

## 📐 Genel Mimari Diyagramı

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           UNet Architecture                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUTS                                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│   │  x (noisy)   │  │      t       │  │ class_label  │                      │
│   │ (B,3,32,32)  │  │    (B,)      │  │    (B,)      │                      │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                      │
│          │                 │                  │                              │
│          │                 ▼                  ▼                              │
│          │          ┌─────────────────────────────┐                         │
│          │          │   Time + Class Embedding    │                         │
│          │          │      (B,) → (B, 256)        │                         │
│          │          └──────────────┬──────────────┘                         │
│          │                         │                                         │
│          ▼                         │ (emb broadcast to all blocks)          │
│   ┌──────────────┐                 │                                        │
│   │   conv_in    │                 │                                        │
│   │  3 → 64 ch   │                 │                                        │
│   └──────┬───────┘                 │                                        │
│          │                         │                                        │
│          ▼                         │                                        │
│   ══════════════════════════════════════════════════                        │
│         ENCODER (Downsampling Path)                                         │
│   ══════════════════════════════════════════════════                        │
│          │                         │                                        │
│          ▼                         │                                        │
│   ┌──────────────┐    skip₁       │                                        │
│   │ ResBlock×2   │◄───────────────┤                                        │
│   │  64 → 64     │                │                                        │
│   └──────┬───────┘                │                                        │
│          ▼                        │                                        │
│   ┌──────────────┐                │                                        │
│   │  Downsample  │  32×32 → 16×16 │                                        │
│   └──────┬───────┘                │                                        │
│          ▼                        │                                        │
│   ┌──────────────┐    skip₂       │                                        │
│   │ ResBlock×2   │◄───────────────┤                                        │
│   │  64 → 128    │                │                                        │
│   ├──────────────┤                │                                        │
│   │ Attention?   │ (if res=16)    │                                        │
│   └──────┬───────┘                │                                        │
│          ▼                        │                                        │
│   ┌──────────────┐                │                                        │
│   │  Downsample  │  16×16 → 8×8   │                                        │
│   └──────┬───────┘                │                                        │
│          ▼                        │                                        │
│   ┌──────────────┐    skip₃       │                                        │
│   │ ResBlock×2   │◄───────────────┤                                        │
│   │ 128 → 256    │                │                                        │
│   ├──────────────┤                │                                        │
│   │ Attention    │ (res=8 ✓)      │                                        │
│   └──────┬───────┘                │                                        │
│          │                        │                                        │
│   ══════════════════════════════════════════════════                        │
│         MIDDLE (Bottleneck)                                                 │
│   ══════════════════════════════════════════════════                        │
│          │                        │                                        │
│          ▼                        │                                        │
│   ┌──────────────┐                │                                        │
│   │  ResBlock    │◄───────────────┤ (emb)                                  │
│   │  256 → 256   │                │                                        │
│   ├──────────────┤                │                                        │
│   │ Attention    │                │                                        │
│   ├──────────────┤                │                                        │
│   │  ResBlock    │◄───────────────┤ (emb)                                  │
│   │  256 → 256   │                │                                        │
│   └──────┬───────┘                │                                        │
│          │                        │                                        │
│   ══════════════════════════════════════════════════                        │
│         DECODER (Upsampling Path)                                           │
│   ══════════════════════════════════════════════════                        │
│          │                        │                                        │
│          ▼                        │                                        │
│   ┌──────────────┐                │                                        │
│   │  cat(h,skip) │◄───────────────┴── skip₃                                │
│   │ ResBlock ×3  │                                                         │
│   │ 256+256→256  │                                                         │
│   ├──────────────┤                                                         │
│   │ Attention    │                                                         │
│   └──────┬───────┘                                                         │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │   Upsample   │  8×8 → 16×16                                            │
│   └──────┬───────┘                                                         │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │  cat(h,skip) │◄─────────────────── skip₂                               │
│   │ ResBlock ×3  │                                                         │
│   │ 256+128→128  │                                                         │
│   └──────┬───────┘                                                         │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │   Upsample   │  16×16 → 32×32                                          │
│   └──────┬───────┘                                                         │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │  cat(h,skip) │◄─────────────────── skip₁                               │
│   │ ResBlock ×3  │                                                         │
│   │ 128+64 → 64  │                                                         │
│   └──────┬───────┘                                                         │
│          │                                                                  │
│   ══════════════════════════════════════════════════                        │
│         OUTPUT                                                              │
│   ══════════════════════════════════════════════════                        │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │  GroupNorm   │                                                         │
│   │    SiLU      │                                                         │
│   │  conv_out    │  64 → 3                                                 │
│   └──────┬───────┘                                                         │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────┐                                                         │
│   │   OUTPUT     │                                                         │
│   │ (B,3,32,32)  │  ← predicted noise (DDPM) or velocity (Flow Matching)   │
│   └──────────────┘                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Bileşenlerin Detaylı Açıklaması

### 1. Time Embedding (Sinusoidal Position Embedding)

**Amaç:** Model'a "şu an hangi t'deyiz" bilgisini vermek.

**Neden önemli?**
- `t=0`: Görüntü neredeyse clean, az noise var
- `t=999`: Pure Gaussian noise, model her şeyi tahmin etmeli
- Bu bilgi olmadan model farklı noise seviyelerini ayırt edemez!

**Formül:**
```
PE(t, 2i) = sin(t / 10000^(2i/dim))
PE(t, 2i+1) = cos(t / 10000^(2i/dim))
```

**Shape akışı:**
```
Input:  t = (B,)           # Batch of timesteps, e.g., [0, 500, 999, 100]
        ↓
Sinusoidal: (B, dim)       # e.g., (4, 256)
        ↓
MLP: Linear → SiLU → Linear
        ↓
Output: (B, dim)           # e.g., (4, 256) - ready to broadcast
```

**Kod:**
```python
# embeddings.py
class SinusoidalPositionEmbedding(nn.Module):
    def forward(self, timesteps):
        # timesteps: (B,) → (B, dim)
        freqs = exp(-log(10000) * arange(dim/2) / (dim/2))
        args = timesteps[:, None] * freqs[None, :]
        return cat([sin(args), cos(args)], dim=-1)
```

---

### 2. Class Embedding (nn.Embedding)

**Amaç:** Model'a "hangi sınıfı üretmeli" bilgisini vermek.

**Neden önemli?**
- Unconditional: Rastgele CIFAR-10 görüntüsü üretir
- Conditional: "Bana bir KEDİ üret" diyebilirsin

**Shape akışı:**
```
Input:  class_label = (B,)     # e.g., [0, 5, 3, 9] (cat, dog, bird, truck)
        ↓
nn.Embedding(10, dim)
        ↓
Output: (B, dim)               # e.g., (4, 256)
```

**Time + Class kombinasyonu:**
```python
t_emb = time_embed(t)         # (B, 256)
c_emb = class_embed(c)        # (B, 256)
combined = t_emb + c_emb      # (B, 256) - elementwise addition
```

---

### 3. ResidualBlock

**Amaç:** Feature extraction + time conditioning

**Neden residual?**
- Gradient akışını kolaylaştırır
- Derin ağlarda training stabilitesi sağlar
- `h + skip` formülü: öğrenilecek şey "fark" olur

**Yapı:**
```
┌────────────────────────────────────────┐
│              ResidualBlock             │
├────────────────────────────────────────┤
│                                        │
│  x ──────────────────────────────┐     │
│  │                               │     │
│  ▼                               │     │
│  GroupNorm → SiLU → Conv3×3      │     │
│  │                               │     │
│  ▼                               │     │
│  + t_emb[:,:,None,None] ◄────────┤     │
│  │                     broadcast │     │
│  ▼                               │     │
│  GroupNorm → SiLU → Dropout      │     │
│  │                               │     │
│  ▼                               │     │
│  Conv3×3                         │     │
│  │                               │     │
│  ▼                               │     │
│  + ◄─────────────────────────────┘     │
│  │         skip connection             │
│  ▼                                     │
│  output                                │
│                                        │
└────────────────────────────────────────┘
```

**Shape akışı:**
```
Input:  x = (B, C_in, H, W)      # e.g., (4, 64, 32, 32)
        t_emb = (B, dim)          # e.g., (4, 256)
        ↓
Conv3×3: (B, C_in, H, W) → (B, C_out, H, W)
        ↓
+ t_emb[:,:,None,None]: time conditioning broadcast
        ↓
Conv3×3: keeps spatial size
        ↓
+ skip_conv(x): residual connection
        ↓
Output: (B, C_out, H, W)         # e.g., (4, 128, 32, 32)
```

---

### 4. Downsample / Upsample

**Amaç:** Spatial resolution değiştirmek

**Downsample (Encoder):**
```
Input:  (B, C, 32, 32)
        ↓
Conv2d(stride=2, kernel=3, padding=1)
        ↓
Output: (B, C, 16, 16)   # Spatial size halved
```

**Upsample (Decoder):**
```
Input:  (B, C, 16, 16)
        ↓
F.interpolate(scale_factor=2, mode='nearest')
        ↓
Conv2d(kernel=3, padding=1)
        ↓
Output: (B, C, 32, 32)   # Spatial size doubled
```

---

### 5. Self-Attention

**Amaç:** Global dependencies yakalamak

**Neden gerekli?**
- Convolution sadece lokal (3×3 veya 5×5 neighborhood)
- Uzaktaki pikseller arasındaki ilişkiyi göremez
- Attention: "Bu piksel, şu uzaktaki pikselle ne kadar ilişkili?"

**Nerede kullanılır?**
- Düşük resolution'larda (8×8, 16×16)
- Yüksek resolution'da çok pahalı: O(N²) where N = H×W

**Yapı:**
```
┌────────────────────────────────────────┐
│            Self-Attention              │
├────────────────────────────────────────┤
│                                        │
│  x ──────────────────────────────┐     │
│  │                               │     │
│  ▼                               │     │
│  GroupNorm                       │     │
│  │                               │     │
│  ▼                               │     │
│  Conv1×1 → Q, K, V               │     │
│  │                               │     │
│  ▼                               │     │
│  Reshape: (B,C,H,W) → (B,heads,N,d)    │
│  │         where N = H×W         │     │
│  ▼                               │     │
│  Attention = softmax(Q·K^T / √d) │     │
│  │                               │     │
│  ▼                               │     │
│  Output = Attention · V          │     │
│  │                               │     │
│  ▼                               │     │
│  Reshape: (B,heads,N,d) → (B,C,H,W)    │
│  │                               │     │
│  ▼                               │     │
│  Conv1×1 (projection)            │     │
│  │                               │     │
│  ▼                               │     │
│  + ◄─────────────────────────────┘     │
│  │         residual                    │
│  ▼                                     │
│  output                                │
│                                        │
└────────────────────────────────────────┘
```

**Shape akışı:**
```
Input:  x = (B, C, H, W)         # e.g., (4, 256, 8, 8)
        ↓
Reshape: (B, C, 64) → (B, heads, 64, head_dim)
        ↓
Q, K, V projections
        ↓
Attention: (B, heads, 64, 64)    # N×N attention matrix
        ↓
Apply to V: (B, heads, 64, head_dim)
        ↓
Reshape back: (B, C, 8, 8)
        ↓
Output: (B, C, H, W)             # e.g., (4, 256, 8, 8) - same shape!
```

---

### 6. Skip Connections

**Amaç:** Encoder'dan decoder'a bilgi aktarmak

**Neden önemli?**
- Downsampling sırasında fine details kayboluyor
- Skip connections bu detayları korur
- "U" şeklini oluşturan yapı bu!

**Görsel:**
```
ENCODER                              DECODER
────────                             ────────
[64ch, 32×32] ─────────────────────► cat([h, skip]) → ResBlock
      ↓                                    ↑
   Downsample                          Upsample
      ↓                                    ↑
[128ch, 16×16] ────────────────────► cat([h, skip]) → ResBlock
      ↓                                    ↑
   Downsample                          Upsample
      ↓                                    ↑
[256ch, 8×8] ──────────────────────► cat([h, skip]) → ResBlock
      ↓                                    ↑
      └───────► MIDDLE ────────────────────┘
```

**Channel concatenation:**
```python
# Decoder'da her ResBlock'tan önce:
h = (B, 256, 8, 8)      # Current hidden state
skip = (B, 256, 8, 8)   # From encoder
h = torch.cat([h, skip], dim=1)  # → (B, 512, 8, 8)
h = resblock(h, emb)    # → (B, 256, 8, 8)
```

---

## 📊 Full Forward Pass: Shape Tracking

```python
# Example: B=4, CIFAR-10 (32×32), base_channels=64, channel_mults=(1,2,4)

# INPUTS
x = (4, 3, 32, 32)       # Noisy image
t = (4,)                 # Timesteps [100, 500, 800, 300]
c = (4,)                 # Classes [0, 5, 3, 9]

# EMBEDDINGS
emb = time_class_embed(t, c)   # → (4, 256)

# INITIAL CONV
h = conv_in(x)                 # (4, 3, 32, 32) → (4, 64, 32, 32)
skips = [h]                    # Store for later

# ENCODER
# Level 0: 64 channels, 32×32
h = resblock(h, emb)           # (4, 64, 32, 32) → (4, 64, 32, 32)
skips.append(h)
h = resblock(h, emb)           # (4, 64, 32, 32) → (4, 64, 32, 32)
skips.append(h)
h = downsample(h)              # (4, 64, 32, 32) → (4, 64, 16, 16)
skips.append(h)

# Level 1: 128 channels, 16×16
h = resblock(h, emb)           # (4, 64, 16, 16) → (4, 128, 16, 16)
skips.append(h)
h = resblock(h, emb)           # (4, 128, 16, 16) → (4, 128, 16, 16)
skips.append(h)
h = downsample(h)              # (4, 128, 16, 16) → (4, 128, 8, 8)
skips.append(h)

# Level 2: 256 channels, 8×8
h = resblock(h, emb)           # (4, 128, 8, 8) → (4, 256, 8, 8)
h = attention(h)               # (4, 256, 8, 8) → (4, 256, 8, 8) ← Attention!
skips.append(h)
h = resblock(h, emb)           # (4, 256, 8, 8) → (4, 256, 8, 8)
h = attention(h)               # (4, 256, 8, 8) → (4, 256, 8, 8)
skips.append(h)

# MIDDLE (Bottleneck)
h = mid_resblock1(h, emb)      # (4, 256, 8, 8) → (4, 256, 8, 8)
h = mid_attention(h)           # (4, 256, 8, 8) → (4, 256, 8, 8)
h = mid_resblock2(h, emb)      # (4, 256, 8, 8) → (4, 256, 8, 8)

# DECODER
# Level 2 → Level 1
skip = skips.pop()             # (4, 256, 8, 8)
h = cat([h, skip], dim=1)      # (4, 256, 8, 8) + (4, 256, 8, 8) → (4, 512, 8, 8)
h = resblock(h, emb)           # (4, 512, 8, 8) → (4, 256, 8, 8)
h = attention(h)
# ... repeat for all skips
h = upsample(h)                # (4, 256, 8, 8) → (4, 256, 16, 16)

# ... continues until original resolution ...

# OUTPUT
h = norm_out(h)                # GroupNorm
h = silu(h)                    # Activation
h = conv_out(h)                # (4, 64, 32, 32) → (4, 3, 32, 32)

# FINAL OUTPUT
output = (4, 3, 32, 32)        # Same shape as input!
                               # This is ε (noise) for DDPM
                               # or v (velocity) for Flow Matching
```

---

## 🎯 Özet: Her Bileşenin Rolü

| Bileşen | Girdi | Çıktı | Rolü |
|---------|-------|-------|------|
| **Time Embed** | `(B,)` | `(B, dim)` | Noise seviyesi bilgisi |
| **Class Embed** | `(B,)` | `(B, dim)` | Sınıf conditioning |
| **conv_in** | `(B,3,H,W)` | `(B,C,H,W)` | Channel projection |
| **ResBlock** | `(B,C,H,W)` | `(B,C',H,W)` | Feature extraction + t cond. |
| **Downsample** | `(B,C,H,W)` | `(B,C,H/2,W/2)` | Resolution azalt |
| **Attention** | `(B,C,H,W)` | `(B,C,H,W)` | Global relationships |
| **Middle** | `(B,C,H,W)` | `(B,C,H,W)` | Bottleneck processing |
| **Upsample** | `(B,C,H,W)` | `(B,C,2H,2W)` | Resolution artır |
| **Skip cat** | `h + skip` | `concat(h,skip)` | Fine details koru |
| **conv_out** | `(B,C,H,W)` | `(B,3,H,W)` | Final prediction |

---

## 💡 Key Insights

1. **Input = Output boyutu:** `(B, 3, 32, 32) → (B, 3, 32, 32)`
   - Model, görüntü ile aynı boyutta bir "şey" tahmin eder
   - DDPM: noise ε
   - Flow Matching: velocity v

2. **t embedding her ResBlock'a gider:**
   - Her layer'da "şu an hangi t'deyiz" bilgisi var
   - Bu sayede model t=0 vs t=999 farkını öğrenir

3. **Attention sadece düşük resolution'da:**
   - 8×8 = 64 token → 64×64 = 4096 attention hesabı (OK)
   - 32×32 = 1024 token → 1024×1024 = 1M attention hesabı (TOO SLOW)

4. **Skip connections kritik:**
   - Onsuz model çok kötü performans gösterir
   - Fine spatial details encoder'dan decoder'a aktarılır
