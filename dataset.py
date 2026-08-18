from torchvision.datasets import VOCDetection
from torch.utils.data import Subset
import os
from torch.utils.data import random_split
from torchvision.transforms import transforms
voc_root = "./data"

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
    )                         # veri seti cıktısını dictten grid matrisine cevircez
    test_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
class Dataset():
    def __init__(self):
        def encode_target(self):
            pass

if __name__ == "__main__":
    train_data, test_data = data_load(voc_root=voc_root)
    image, target = train_data[0]