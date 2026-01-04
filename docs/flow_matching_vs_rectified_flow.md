# Flow Matching vs. Rectified Flow (Reflow)

Bu doküman, projemizde kullandığımız **Conditional Flow Matching (CFM)** ile onun daha gelişmiş bir versiyonu olan **Rectified Flow (Reflow)** arasındaki farkları, teorik temelleri ve matematiksel geçişi açıklar.

---

## 1. Temel Problem: "Trajectory" (Yol) Düz mü?

Görüntü üretiminde amaç, bir gürültü dağılımından ($x_0 \sim \mathcal{N}(0, I)$) veri dağılımına ($x_1 \sim \mathcal{D}$) bir haritalama (mapping) öğrenmektir.

### Conditional Flow Matching (Bizim Kullandığımız)
Bizim şu anki implementasyonumuzda, her veri noktası $x_1$ için rastgele bir gürültü $x_0$ seçiyoruz ve aralarına dümdüz bir çizgi çekiyoruz:

$$
x_t = (1 - t)x_0 + t x_1
$$

Bu **"Conditional Probability Path"**tir. Modelimiz bu path üzerindeki hız alanını (velocity field) öğrenir:
$$
v_t(x_t) = x_1 - x_0
$$

**Sorun:**
Bireysel path'ler (bir gürültü - bir resim arası) düz çizgi olsa da, modelin öğrendiği **Global Velocity Field** (tüm resimlerin ortalaması) düz çizgiler oluşturmaz.
Eğitim sırasında rastgele ($x_0, x_1$) çiftleri kullandığımız için (Independent Coupling), farklı path'ler birbiriyle kesişir (crossing paths).

**Sonuç:**
Çaprazlaşan yollar nedeniyle, integral alarak (ODE solver ile) $x_0$'dan $x_1$'e gitmek zordur. Bu yüzden 20-50 adım gerekir. Tek adımda gitmeye çalışırsanız (Euler step, $\Delta t=1$) çok büyük hata yaparsınız.

---

## 2. Rectified Flow (Reflow) Nedir?

Rectified Flow, bu "kesişen" yolları **dümdüz ve paralel** hale getirme işlemidir.

**Temel Fikir:**
Eğer elimizde $x_0$ ve $x_1$ çiftleri öyle eşleştirilmiş olsaydı ki, ODE yörüngesi gerçekten dümdüz olsaydı, o zaman $x_1 = x_0 + v(x_0)$ diyebilirdik ve **TEK ADIMDA** (One-Step) üretim yapabilirdik.

Reflow işlemi iteratif bir süreçtir:

### Matematiksel Prosedür (Reflow)

Buna **Recitification** (Doğrultma) denir.

**Adım 0 (Bizim Yaptığımız):**
Rastgele $(x_0, x_1)$ çiftleri ile bir model eğit ($\phi_1$). Bu modelin ürettiği yörüngeler biraz kavisli (curved).

**Adım 1 (1-Rectified Flow):**
Eğitilmiş model $\phi_1$'i al. Bu modelle gürültüden görüntü üret:
$$
z_0 \sim \mathcal{N}(0, I) \xrightarrow{\phi_1} z_1
$$
Artık elimizde yeni bir çift var: $(z_0, z_1)$.
Bu çiftin özelliği şu: $z_0$ ve $z_1$ rastgele değil, modelin kendi dinamiğiyle birbirine bağlandı.

**Adım 2 (Yeniden Eğitim):**
Bu yeni $(z_0, z_1)$ çiftlerini kullanarak **yeni bir model ($\phi_2$) eğit**.
Yine aynı loss fonksiyonu:
$$
\mathcal{L}(\theta) = \mathbb{E}_{t, (z_0, z_1)} \| v_\theta(x_t) - (z_1 - z_0) \|^2
$$

**Sonuç:**
Yeni model $\phi_2$'nin learned vector field'ı çok daha düz (straight) olur. Çünkü artık eğitim verisi, modelin kendi "tercih ettiği" akış yolu üzerindedir.

Bu işlemi teorik olarak sonsuza kadar tekrarlayabilirsiniz ($k$-Rectified Flow), ama genellikle 1 kere yapmak (1-Rectified) harika sonuç verir.

----

## 3. Karşılaştırma Tablosu

| Özellik | Standard Flow Matching (Bizimki) | Rectified Flow (Reflow) |
|---------|----------------------------------|-------------------------|
| **Coupling** | Independent (Rastgele) | Learned (Model Üretimi) |
| **Trajectory** | Kesişen, kavisli (Curved) | Düz, paralel (Straight) |
| **Eğitim Süreci** | 1 aşamalı | 2 veya daha fazla aşamalı |
| **Sampling Steps** | 20 - 50 adım | **1 - 2 adım** (One-Step) |
| **Maliyet** | Standart | 2x Eğitim maliyeti |
| **Matematik** | $v = x_1 - x_0$ (Data) | $v = z_1 - z_0$ (Model) |

---

## 4. Neden "Rectified"?

İsim matematikteki "Rectifiable curve" (uzunluğu ölçülebilir eğri) veya geometrideki "Rectification" (bir eğriyi düz çizgiye dönüştürme) kavramından gelir.

Bizim bağlamımızda:
> "Transport cost" (taşıma maliyetini) minimize etmek için yolları kısaltıyoruz. En kısa yol düz bir çizgidir. Reflow işlemi, kavisli diffusion yollarını düz çizgilere "doğrultur".

---

## 5. Kodda Ne Değişirdi?

Eğer projemize Reflow ekleseydik:

1.  **Mevcut kodumuz kalırdı.** İlk model (`base_model`) olarak eğitilirdi.
2.  **Generate Dataset Scripti:**
    *   Bu `base_model` kullanılarak 50,000 tane $(z_0, z_1)$ çifti oluşturulup diske kaydedilirdi. (`z_0`: noise, `z_1`: generated image).
3.  **Reflow Training:**
    *   `train.py` aynı kalırdı!
    *   Sadece `Dataset` sınıfı değiştirilip, CIFAR-10 yerine bizim kaydettiğimiz bu "sentetik" $(z_0, z_1)$ çiftleri yüklenirdi.
4.  **Sampling:**
    *   Artık `steps=1` ile Euler solver kullanarak mükemmel sonuç alırdık.

---

## 6. Referanslar

*   [Flow Straight and Fast: Learning to Generate with One-Step Flow Matching](https://arxiv.org/abs/2210.02747) (Orijinal Flow Matching)
*   [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
*   [Rectified Flow on ICLR 2023](https://arxiv.org/abs/2209.03003) (Liu et al.) - **Reflow Makalesi**
