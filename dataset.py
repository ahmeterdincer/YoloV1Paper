from torchvision.datasets import VOCDetection
from torch.utils.data import Subset
import os

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
    train_data_cutted = Subset(train_dataset, range(int(len(train_dataset)*0.6)))
    test_data_cutted = Subset(test_dataset, range(int(len(test_dataset)*0.15)))
    print("Cutted sets")
    print(len(train_data_cutted))
    print(len(test_data_cutted))
if __name__ == "__main__":
    data_load(voc_root=voc_root)