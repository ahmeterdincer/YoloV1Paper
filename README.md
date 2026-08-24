# 🎯 YOLOv1 (You Only Look Once) — Scratch PyTorch Implementation

Bu proje, Joseph Redmon ve ekibi tarafından 2015 yılında yayınlanan **"You Only Look Once: Unified, Real-Time Object Detection"** makalesinin, hiçbir hazır kütüphane (Ultralytics vb.) kullanılmadan, **saf PyTorch (from scratch)** ile adım adım koda dökülmüş eksiksiz bir uygulamasıdır.

Veri seti olarak **Pascal VOC 2007** (20 sınıf) kullanılmıştır.

---

## 📌 Projenin Öne Çıkan Özellikleri ve El Emeği Modüller

Bu projede derin öğrenme mimarisinin her bir bileşeni makaleye sadık kalınarak sıfırdan inşa edilmiştir:

1. **Özel Target Encoding Matrisi ($7 \times 7 \times 30$):** Pascal VOC XML etiketlerini okuyup her nesneyi ilgili $7 \times 7$ grid hücresine ve hücre içi göreli $[x, y, w, h]$ koordinatlarına dönüştüren özel kodlama.
2. **24 Katmanlı Orijinal Darknet Mimarisi:** Makaledeki *Table 1* referans alınarak $1 \times 1$ ve $3 \times 3$ evrişim katmanlarıyla inşa edilmiş CNN ve Fully Connected kafa.
3. **5 Parçalı Özel YOLOv1 Kayıp Fonksiyonu (YOLOLoss):** Sorumlu kutu (Responsible Box) seçim mekanizması, $\lambda_{coord}=5.0$ ve $\lambda_{noobj}=0.5$ ağırlıkları ile SSE (Sum-Squared Error) hesaplaması.
4. **Sıfırdan IoU ve NMS (Non-Maximum Suppression):** Model çıktısındaki mükerrer kutuları eleyen saf Python/PyTorch NMS algoritması.
5. **Görselleştirme ve Çizim Modülü:** Normalize YOLO koordinatlarını piksel koordinatlarına dönüştüren ve sınıflarıyla birlikte ekrana çizen Matplotlib motoru.

---

## 🧱 Proje Mimarisi ve Dosya Düzeni

```text
YoloV1Paper/
├── data/              # Pascal VOC 2007 veri seti (Otomatik indirilir)
├── paper/             # Orijinal YOLOv1 makalesi (PDF)
├── config.py          # Hiperparametreler, sabitler ve Pascal VOC sınıf listesi
├── utils.py           # IoU hesabı, NMS algoritması ve Bounding Box çizim fonksiyonu
├── dataset.py         # Pascal VOC veri yükleyici, Veri Artırma ve Target Encoding
├── model.py           # 24 Conv katmanlı YOLOv1 CNN mimarisi
├── loss.py            # 5 parçalı özel YOLOv1 kayıp fonksiyonu (YOLOLoss)
├── engine.py          # Train ve Test döngüsü fonksiyonları (tqdm destekli)
├── train.py           # Eğitimi başlatan ve modeli kaydeden ana script
├── predict.py         # Test görselinde nesneleri tespit edip çizen tahmin modülü
└── README.md          # Proje dokümantasyonu
```

---

## 🔬 Matematiksel ve Teknik Detaylar

### 1. Çıktı Tensörü Anatomisi: $(Batch, 7, 7, 30)$
Görsel $7 \times 7 = 49$ ızgaraya (grid) bölünür. Her hücre $30$ uzunluğunda bir vektör tahmin eder:
- **$[0:20]$ (20 Eleman):** 20 Pascal VOC sınıfının olasılıkları ($P(\text{Class}_i | \text{Object})$)
- **$[20]$ (1 Eleman):** 1. Kutunun Güven Skoru ($C_1$)
- **$[21:25]$ (4 Eleman):** 1. Kutunun Koordinatları $[x_{cell}, y_{cell}, w, h]$
- **$[25]$ (1 Eleman):** 2. Kutunun Güven Skoru ($C_2$)
- **$[26:30]$ (4 Eleman):** 2. Kutunun Koordinatları $[x_{cell}, y_{cell}, w, h]$

---

### 2. YOLOv1 Kayıp Fonksiyonu (5 Parça)

$$\mathcal{L}_{YOLO} = \mathcal{L}_{xy} + \mathcal{L}_{wh} + \mathcal{L}_{obj} + \mathcal{L}_{noobj} + \mathcal{L}_{class}$$

* **Konum Kaybı ($x, y$):** Sorumlu seçilen kutunun merkez sapması ($\lambda_{coord} = 5.0$ ile çarpılır).
* **Boyut Kaybı ($w, h$):** Küçük kutulardaki hataları daha sert cezalandırmak için karekök farkı alınır: $(\sqrt{w} - \sqrt{\hat{w}})^2$ ($\lambda_{coord} = 5.0$). Gradyan çökmesini engellemek için `torch.sqrt(torch.abs(w) + 1e-6)` kullanılmıştır.
* **Nesne Güven Kaybı ($C_{obj}$):** Yalnızca nesne olan hücredeki en iyi kutunun ($1.0$'a yakınsama) hatası.
* **Boş Hücre Güven Kaybı ($C_{noobj}$):** Nesne olmayan 48 arkaplan hücresindeki her iki kutunun güven skorunu $0$'a çekme hatası ($\lambda_{noobj} = 0.5$).
* **Sınıflandırma Kaybı ($P_{class}$):** 20 sınıfın One-Hot karesel hatası.

---

### 3. Sorumlu Kutu Seçimi (Responsible Box Selection)
Bir hücrede nesne varsa, model o hücre için 2 farklı kutu ($B=2$) tahmin eder. `utils.py` içindeki `iou()` fonksiyonu ile gerçek kutuya en yüksek IoU örtüşmesini sağlayan kutu `argmax` ile **"Sorumlu Kutu"** olarak atanır ve koordinat cezaları yalnızca bu kutu üzerinden kesilir.

---

## ⚙️ Kurulum ve Gereksinimler

Gerekli kütüphaneleri yüklemek için:

```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

---

## 🚀 Kullanım

### 1. Modeli Eğitmek
Modeli Pascal VOC 2007 veri seti üzerinde eğitmek için:

```bash
python train.py
```
*Bu komut veri setini `./data` klasörüne otomatik olarak indirir, eğitimi başlatır ve en iyi model ağırlıklarını `yolov1.pth` olarak kaydeder.*

### 2. Test ve Tahmin (Inference)
Eğitilmiş modeli test veri setinden rastgele bir görsel üzerinde denemek ve ekranda etiketli kutuları görmek için:

```bash
python predict.py
```

---

## 📊 Pascal VOC Sınıfları (20 Sınıf)
`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.

---

## 📜 Kaynakça
* Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
