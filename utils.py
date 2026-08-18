# [Adım 2] utils.py (Temel)   -> IoU ve Bounding Box çizim fonksiyonlarını yaz
def box_boundaries( xmin:float, ymin:float, xmax:float, ymax:float):
    x_center= (xmin+xmax)/2
    y_center = (ymin+ymax)/2
    w = xmax-xmin
    h = ymax-ymin
    return x_center, y_center, w,h

def max(a,b):
    if a>b: return a
    return b

def min(a,b):
    if a<b: return a
    return b

def iou(real_list: list, pred_list: list):
    # [x, y, w, h] merkez koordinatlarını ve boyutları al
    x1, y1, w1, h1 = real_list[0], real_list[1], real_list[2], real_list[3]
    x2, y2, w2, h2 = pred_list[0], pred_list[1], pred_list[2], pred_list[3]

    # 1. Adım: Merkez koordinatlarından [x_min, y_min, xmax, ymax] köşe koordinatlarına geçiş
    box1_x1 = x1 - (w1 / 2)
    box1_y1 = y1 - (h1 / 2)
    box1_x2 = x1 + (w1 / 2)
    box1_y2 = y1 + (h1 / 2)

    box2_x1 = x2 - (w2 / 2)
    box2_y1 = y2 - (h2 / 2)
    box2_x2 = x2 + (w2 / 2)
    box2_y2 = y2 + (h2 / 2)

    # 2. Adım: Kesişim Kutusunun Sol-Üst (max) ve Sağ-Alt (min) Köşeleri
    x1_inter = max(box1_x1, box2_x1)
    y1_inter = max(box1_y1, box2_y1)
    x2_inter = min(box1_x2, box2_x2)
    y2_inter = min(box1_y2, box2_y2)
    
    # 3. Adım: Kesişim genişlik ve yüksekliği (negatif olamaz)
    w_inter = max(0, x2_inter - x1_inter)
    h_inter = max(0, y2_inter - y1_inter)
    
    # Kesişim Alanı
    intersection_area = w_inter * h_inter   
    
    # 4. Adım: Kutuların Bireysel Alanları
    area_a = w1 * h1
    area_b = w2 * h2
    
    # 5. Adım: Birleşim Alanı (Union Area)
    union_area = area_a + area_b - intersection_area
    
    # 6. Adım: IoU Hesabı (Sıfıra bölme hatasına karşı epsilon eklendi)
    iou_score = intersection_area / (union_area + 1e-6)
    
    return iou_score