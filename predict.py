from model import YOLOv1
import torch
import dataset
from config import VOC_ROOT
import random
model= YOLOv1()
model.state_dict(torch.load("yolov1.pth"))

_, test_data = dataset.data_load(VOC_ROOT)
random_idx = random.randint(0, len(test_data)-1)
image, target = test_data[random_idx]

_,test_transforms= dataset.feature_extracture()
image_tensor = test_transforms(image).unsqueeze(0)

model.eval()
with torch.no_grad():
    prediction = model(image_tensor)