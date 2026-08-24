from torchvision.datasets import VOCDetection
import torch
from torch.utils.data import Subset
import os
from torch.utils.data import random_split
from torchvision.transforms import transforms,ToTensor
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


def data_load(voc_root):
    os.makedirs(voc_root, exist_ok=True)
    
    train_dataset = VOCDetection(
        root=voc_root,
        year='2007',
        image_set='trainval',
        download=True
    )

    test_dataset = VOCDetection(
        root=voc_root,
        year='2007',
        image_set='test',
        download=True
    )
    
    print(f"train length: {len(train_dataset)}\n test length: {len(test_dataset)}")
    
    train_data_cutted, _ = random_split(train_dataset, lengths=[0.6,0.4])
    test_data_cutted, _ = random_split(test_dataset, lengths=[0.15,0.85])
    print("Cutted sets")
    print(len(train_data_cutted))
    print(len(test_data_cutted))
    return train_data_cutted, test_data_cutted

def feature_extracture():
    train_transforms = transforms.Compose(
        [
            transforms.Resize((448,448)),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.2,0.2),
                scale=(0.8,1.2)),
            transforms.ColorJitter(
                brightness=0.2,
                saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            
        ]
    ) 
    test_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, test_transforms
from config import S, B, C, VOC_CLASSES

def encode_target(image, target, S=S, B=B, C=C):

    image_width, image_height = image.size
    
    target_matrix = torch.zeros((S, S, C + B * 5))
    
    objects = target['annotation']['object']
    if isinstance(objects, dict):
        objects = [objects]
        
    for obj in objects:
        class_name = obj['name']
        if class_name not in VOC_CLASSES:
            continue
            
        class_idx = VOC_CLASSES.index(class_name)

        xmin = float(obj['bndbox']['xmin']) / image_width
        ymin = float(obj['bndbox']['ymin']) / image_height
        xmax = float(obj['bndbox']['xmax']) / image_width
        ymax = float(obj['bndbox']['ymax']) / image_height
        
        # Merkez (x, y) ve Boyut (w, h) hesapla
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin
        
        # Hangi grid hücresine (satır, sütun) düştüğünü bul
        row = int(y_center * S)
        col = int(x_center * S)
        
        # Sınır taşmalarını engelle (0 ile S-1 arası)
        row = min(row, S - 1)
        col = min(col, S - 1)
        
        # Hücre içi göreli koordinatlar (0 ile 1 arası)
        x_cell = x_center * S - col
        y_cell = y_center * S - row
        
        # Eğer o hücreye daha önce nesne atanmamışsa (1. kutunun Confidence skoru 0 ise)
        if target_matrix[row, col, C] == 0:
            # 1. Sınıf One-Hot kodlaması (0 - 19)
            target_matrix[row, col, class_idx] = 1.0
            
            # 2. 1. Kutunun Confidence Skoru (İndeks 20)
            target_matrix[row, col, C] = 1.0
            
            # 3. 1. Kutunun Konumu [x_cell, y_cell, w, h] (İndeks 21 - 24)
            target_matrix[row, col, C + 1 : C + 5] = torch.tensor([x_cell, y_cell, width, height])
            
    return target_matrix

class Dataset(Dataset):
    def __init__(self,dataset: Dataset, S:int, B:int, C:int, transform=None):
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        image, target = self.dataset[index]
        target_matrix = encode_target(image=image, target=target, S=self.S, B=self.B, C=self.C)

        if self.transform:
            image=self.transform(image)
        
        return image, target_matrix



if __name__ == "__main__":
    train = data_load("./data")
    image, target = train[0]
    target_matrix = encode_target(image, target)
    print("Target Matrix Shape:", target_matrix.shape)
    print("Nesne içeren hücrelerin koordinatları:")
    for r in range(S):
        for c in range(S):
            if target_matrix[r, c, C] == 1.0:
                cls_idx = torch.argmax(target_matrix[r, c, :C]).item()
                box = target_matrix[r, c, C+1:C+5].tolist()
                print(f"-> Hücre ({r}, {c}): Sınıf='{VOC_CLASSES[cls_idx]}', Kutu={box}")
