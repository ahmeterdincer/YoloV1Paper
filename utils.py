# [Adım 2] utils.py (Temel)   -> IoU ve Bounding Box çizim fonksiyonlarını yaz
import torch
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def box_boundaries(xmin: float, ymin: float, xmax: float, ymax: float):
    """Köşe koordinatlarını [xmin, ymin, xmax, ymax] merkez ve boyutlara [x, y, w, h] dönüştürür."""
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return x_center, y_center, w, h

def max(a, b):
    return a if a > b else b

def min(a, b):
    return a if a < b else b

def iou(real_list: list, pred_list: list):
    """İki kutu [x, y, w, h] arasındaki Kesişim / Birleşim (IoU) örtüşme skorunu hesaplar."""
    
    x1, y1, w1, h1 = real_list[0], real_list[1], real_list[2], real_list[3]
    x2, y2, w2, h2 = pred_list[0], pred_list[1], pred_list[2], pred_list[3]

    box1_x1 = x1 - (w1 / 2)
    box1_y1 = y1 - (h1 / 2)
    box1_x2 = x1 + (w1 / 2)
    box1_y2 = y1 + (h1 / 2)

    box2_x1 = x2 - (w2 / 2)
    box2_y1 = y2 - (h2 / 2)
    box2_x2 = x2 + (w2 / 2)
    box2_y2 = y2 + (h2 / 2)

    x1_inter = max(box1_x1, box2_x1)
    y1_inter = max(box1_y1, box2_y1)
    x2_inter = min(box1_x2, box2_x2)
    y2_inter = min(box1_y2, box2_y2)
    
    w_inter = max(0, x2_inter - x1_inter)
    h_inter = max(0, y2_inter - y1_inter)
    
    intersection_area = w_inter * h_inter   
    
    area_a = w1 * h1
    area_b = w2 * h2
    
    union_area = area_a + area_b - intersection_area
    
    iou_score = intersection_area / (union_area + 1e-6)
    
    return iou_score

def plot_boxes(image, boxes, class_names=None):
    """Görseli ve üzerindeki bounding box'ları sınıf isimleriyle birlikte ekrana çizer."""
    
    if type(image)!= torch.Tensor:
        image=ToTensor()(image)
    
    if image.ndim==4 : image.squeeze(0)
    
    image = image.permute(1,2,0)
    
    if image.requires_grad:
        image = image.detach()
        print("***gradyan takibi kapatıldı***")
    
    if image.is_cuda or image.device.type != "cpu":
        print(f"***device {image.device} dan CPU ya cerilmistir***")
        image = image.cpu()
        
    image = image.numpy()
    
    if image.min() < 0:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    
        image = (image * std) + mean
        
        image = np.clip(image, 0, 1)
    
    height = image.shape[0]
    width = image.shape[1]
    
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    for box in boxes:
        score = None
        label = ""
        
        if len(box) == 6:
            cls_val, score, x, y, w, h = box
        elif len(box) == 5:
            cls_val, x, y, w, h = box
        elif len(box) == 4:
            x, y, w, h = box
            cls_val = None
        else:
            continue

        if cls_val is not None:
            if isinstance(cls_val, (int, float, np.integer)) and class_names is not None:
                cls_idx = int(cls_val)
                label = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
            else:
                label = str(cls_val)

        if score is not None:
            label += f" {score:.2f}"

        box_w = w * width
        box_h = h * height
        upper_left_x = (x - w / 2) * width
        upper_left_y = (y - h / 2) * height

        rect = patches.Rectangle(
            (upper_left_x, upper_left_y),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

        if label:
            ax.text(
                upper_left_x,
                max(0, upper_left_y - 4),
                label,
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="red", edgecolor="red", boxstyle="round,pad=0.2", alpha=0.8)
            )

    plt.axis("off")
    plt.tight_layout()
    plt.show()

from config import CONF_THRESHOLD, IOU_THRESHOLD

def nms(boxes: list, conf_treshold: float, ıou_treshold: float):
    """Çakışan mükerrer kutuları IoU ve güven eşiğine göre eleyerek en iyi kutuları seçer."""
    filtered_boxes = [box for box in boxes if box[0] >= conf_treshold]
    filtered_boxes = sorted(filtered_boxes, key=lambda x: x[0], reverse=True)
    chosen_boxes = []
    
    while filtered_boxes:
        chosen_box = filtered_boxes.pop(0)
        chosen_boxes.append(chosen_box)    
        remaining_boxes = []
        
        for des in filtered_boxes:
            if chosen_box[1] != des[1]:
                remaining_boxes.append(des)
            elif iou(chosen_box[2:], des[2:]) < ıou_treshold:
                remaining_boxes.append(des)

        filtered_boxes = remaining_boxes
        
    return chosen_boxes  