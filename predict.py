from model import YOLOv1
import torch
import dataset
from config import VOC_ROOT
import random
from config import S, B, C, VOC_CLASSES, CONF_THRESHOLD, IOU_THRESHOLD, DEVICE
from utils import nms, plot_boxes
model= YOLOv1()
model.load_state_dict(torch.load("yolov1.pth"))

_, test_data = dataset.data_load(VOC_ROOT)
random_idx = random.randint(0, len(test_data)-1)
image, target = test_data[random_idx]

_,test_transforms= dataset.feature_extracture()
image_tensor = test_transforms(image).unsqueeze(0)

model.eval()
with torch.no_grad():
    prediction = model(image_tensor)
    
    
def decode_predictions(predictions:torch.Tensor, S:int, B:int, C:int):
    all_boxes= []
    for row in range(7):
        for col in range(7):
            
            class_idx = torch.argmax(predictions[row,col, :C]).item()
            class_prob = predictions[row,col,class_idx].item()
            conf1 = predictions[row,col,C].item()
            score1 = conf1*class_prob
            x1 = (col+predictions[row,col,C+1].item())/S
            y1 = (row + predictions[row,col,C+2].item())/S
            w1 = (predictions[row,col,C+3].item())
            h1 = predictions[row,col,C+4].item()
            all_boxes.append([score1, class_idx, x1, y1,w1, h1])

            conf2 = predictions[row,col,C+5].item()
            score2 = conf2*class_prob
            x2 = (col+predictions[row,col,C+6].item())/S
            y2 = (row + predictions[row,col,C+7].item())/S
            w2 = (predictions[row,col,C+8].item())
            h2 = predictions[row,col,C+9].item()
            all_boxes.append([score2, class_idx, x2, y2, w2, h2])
    
    return all_boxes

# 1. Modelin (1, 7, 7, 30) çıktısını squeeze edip (7, 7, 30) olarak kutulara çevir:
decoded_boxes = decode_predictions(prediction.squeeze(0), S, B, C)

# 2. NMS ile mükerrer kutuları temizle:
clean_boxes = nms(decoded_boxes, conf_treshold=CONF_THRESHOLD, ıou_treshold=IOU_THRESHOLD)

# 3. Görseli ve kutuları ekrana çizdir:
plot_boxes(image, clean_boxes, class_names=VOC_CLASSES)
