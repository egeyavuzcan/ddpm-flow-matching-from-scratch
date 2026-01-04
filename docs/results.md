# DDPM vs Flow Matching: Training Results

Bu rapor, aynı UNet modeli ve aynı eğitim koşulları altında **DDPM** ve **Flow Matching** yöntemlerinin karşılaştırmasını içerir.

---

## 🔬 Deney Kurulumu

| Parametre | Değer |
|-----------|-------|
| Model | UNetSmall (~2.7M parametre) |
| Dataset | CIFAR-10 (50K görüntü) |
| Image Size | 32×32 |
| Epochs | 100 |
| Batch Size | 128 |
| Learning Rate | 0.0002 |
| Optimizer | AdamW |

---

## 📊 Training Loss Karşılaştırması

### DDPM
```
Initial Loss:  0.1299
Final Loss:    0.0333
Min Loss:      0.0178 (step 22700)
Improvement:   74% ↓
Trend:         Hala düşüyor (slope: -0.000005)
```

### Flow Matching
```
Initial Loss:  0.4292
Final Loss:    0.1606
Min Loss:      0.1567 (step 38600)
Improvement:   62% ↓
Trend:         Hala düşüyor (slope: -0.000054)
```

### Analiz

| Metrik | DDPM | Flow Matching | Yorum |
|--------|------|---------------|-------|
| Final Loss | 0.033 | 0.161 | DDPM loss daha düşük |
| Görsel Kalite | Noise'lu | Düzgün | **FM çok daha iyi!** |
| Trend | Yavaş düşüyor | Hızlı düşüyor | FM daha hızlı öğreniyor |

---

## 🤔 Neden Flow Matching Daha İyi Sonuç Veriyor?

### 1. Loss Değerleri Yanıltıcı

**DDPM loss daha düşük ama görsel kalite daha kötü. Neden?**

- **DDPM:** Noise tahmin ediyor (`ε_θ`)
- **Flow Matching:** Velocity tahmin ediyor (`v_θ`)

Bu iki değer farklı scale'lerde:
- Noise: `~N(0, 1)` - genellikle küçük değerler
- Velocity: `x_1 - x_0` - daha büyük range

**Sonuç:** Loss değerlerini doğrudan karşılaştırmak anlamsız!

### 2. Sampling Adım Sayısı Farkı

Test sırasında kullanılan adımlar:

| Method | Steps | Per-sample Time |
|--------|-------|-----------------|
| DDPM | 100 | 0.88s |
| Flow Matching | 20 | 0.16s |

**DDPM 1000 adımla eğitildi ama 100 adımla test edildi!**

Bu ciddi bir kalite kaybına neden oluyor çünkü:
- DDPM, 1000 adımlık markov zinciri için optimize edildi
- 100 adımla çalıştırınca aradaki adımlar atlanıyor
- Model bu "atlama"yı kompanse edemiyor

### 3. Flow Matching'in Doğal Avantajı

```
DDPM:          Discrete steps, Markov chain
Flow Matching: Continuous ODE, smooth trajectory
```

**Flow Matching avantajları:**

1. **Linear Path:** `x_t = (1-t)·x_0 + t·x_1`
   - Düz bir çizgi, öğrenmesi kolay
   - Velocity her yerde constant

2. **Flexible Sampling:**
   - Herhangi bir adım sayısıyla çalışabilir
   - 20 adım bile iyi sonuç verir

3. **Smoother Trajectories:**
   - ODE çözümü daha stabil
   - Euler method bile yeterli

### 4. DDPM Neden Başarısız?

```
Training:  t ∈ {0, 1, 2, ..., 999} (1000 discrete steps)
Testing:   t ∈ {0, 10, 20, ..., 990} (100 steps, 10'ar atlıyor)
```

DDPM modeli `t=500`'deki noise'u tahmin etmeyi öğrendi.
Ama `t=500` test sırasında atlanıyor, model `t=490` ve `t=510`'u görüyor.

**Çözüm:** DDPM için ya:
- 1000 adım kullan (çok yavaş)
- DDIM sampler kullan (adaptive)
- Daha düşük timestep'le train et

---

## 📈 Sonuç Görüntüleri

### Flow Matching Örnekleri (İyi Kalite)
![Flow Matching Samples](outputs/comparison/flow_matching_samples.png)

### DDPM Örnekleri (Noise'lu)
![DDPM Samples](outputs/comparison/ddpm_samples.png)

### Class Bazlı Karşılaştırma

Her class için üst satır DDPM, alt satır Flow Matching:

| Class | Karşılaştırma |
|-------|---------------|
| Airplane | ![](outputs/comparison/class_0_airplane.png) |
| Automobile | ![](outputs/comparison/class_1_automobile.png) |
| Cat | ![](outputs/comparison/class_3_cat.png) |
| Dog | ![](outputs/comparison/class_5_dog.png) |

---

## ⏱️ Hız Karşılaştırması

| Metod | Adım | Toplam Süre | Örnek Başına |
|-------|------|-------------|--------------|
| DDPM (100 steps) | 100 | 17.6s | 0.88s |
| Flow Matching (20 steps) | 20 | 3.3s | 0.16s |
| **Speedup** | **5x** | **5.4x** | **5.5x** |

---

## 🎯 Öneriler

### DDPM İyileştirmek İçin:
1. **1000 adım kullan:** `--ddpm_steps 1000` (yavaş ama doğru)
2. **DDIM Sampler ekle:** Daha az adımla iyi sonuç verir
3. **Cosine schedule kullan:** Daha smooth geçişler

### Flow Matching İyileştirmek İçin:
1. **Daha fazla epoch:** 200-300 epoch dene
2. **Heun solver:** `--solver heun` (2x yavaş, daha iyi)
3. **Larger model:** `unet` kullan (`unet_small` yerine)

### Genel:
- **EMA (Exponential Moving Average):** Daha stabil sonuçlar
- **CFG (Classifier-Free Guidance):** Daha keskin görüntüler
- **Learning Rate Scheduling:** Cosine decay

---

## 📝 Sonuç

| Kriter | Kazanan | Sebep |
|--------|---------|-------|
| **Görsel Kalite** | 🏆 Flow Matching | Düzgün, tanınabilir görüntüler |
| **Training Speed** | Berabere | Aynı epoch sayısı |
| **Sampling Speed** | 🏆 Flow Matching | 5.4x daha hızlı |
| **Flexibility** | 🏆 Flow Matching | Herhangi step sayısı çalışır |
| **Theoretical Beauty** | DDPM | Derin matematiksel temeller |

**Sonuç:** Pratik uygulamalar için **Flow Matching** tercih edilmeli.

---

## 📚 Referanslar

1. [DDPM - Ho et al., 2020](https://arxiv.org/abs/2006.11239)
2. [Flow Matching - Lipman et al., 2022](https://arxiv.org/abs/2210.02747)
3. [DDIM - Song et al., 2020](https://arxiv.org/abs/2010.02502)
