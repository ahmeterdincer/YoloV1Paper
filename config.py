import torch

# ─── Pascal VOC 20 Sınıf Listesi ──────────────────────
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ─── YOLOv1 Parametreleri ──────────────────────────────
S = 7                # Görsel 7x7'lik ızgaraya (grid) bölünür
B = 2                # Her grid hücresi 2 adet bounding box tahmin eder
C = len(VOC_CLASSES) # Pascal VOC için 20 sınıf (COCO için 80)
IMG_SIZE = 448       # YOLOv1 orijinal girdi çözünürlüğü
# ─── Loss Ağırlıkları (Makale Bölüm 2.2) ────────────────
LAMBDA_COORD = 5.0   # Koordinat kaybının katsayısı (konum hatalarını cezalandırır)
LAMBDA_NOOBJ = 0.5   # Nesne olmayan hücrelerin güven skoru katsayısı
# ─── Eğitim Ayarları ──────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.0005
EPOCHS = 100
CONF_THRESHOLD = 0.4 # NMS öncesi güven eşiği
IOU_THRESHOLD = 0.5  # NMS çakışma eşiği
