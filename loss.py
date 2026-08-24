import torch
import torch.nn as nn
from utils import iou


class YOLOLoss(nn.Module):
    """Sorumlu kutu seçimi, konum, boyut, güven skoru ve sınıf kayıplarını hesaplayan 5 parçalı YOLOv1 Loss sınıfı."""
    def __init__(self, S: int = 7, B: int = 2, C: int = 20, coord: float = 5.0, noobj: float = 0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.coord = coord
        self.noobj = noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, prediction, target):
        pred_bounding_boxes=[]
        true_bounding_box=None
        exists_box = target[...,20:21]
        
        pred_bounding_boxes.append(prediction[21:25])
        pred_bounding_boxes.append(prediction[26:30])
            
        true_bounding_box = target[21:25]
            
        iou_b1 = iou(true_bounding_box,pred_bounding_boxes[0])
        iou_b2 = iou(true_bounding_box,pred_bounding_boxes[1])
            
        best_box = torch.argmax(torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0), dim=0)
            
        xy_loss = self.mse(pred_bounding_boxes[best_box][:2]*exists_box, true_bounding_box[:2]*exists_box)
        wh_loss = self.mse([torch.sqrt(torch.abs(pred_bounding_boxes[best_box][2]) + 1e-6)*exists_box,torch.sqrt(torch.abs(pred_bounding_boxes[best_box][3]) + 1e-6)*exists_box],
                            [torch.sqrt(torch.abs(true_bounding_box[2]) + 1e-6)*exists_box,torch.sqrt(torch.abs(true_bounding_box[3]) + 1e-6)*exists_box])
            
        box_loss = self.coord * (xy_loss + wh_loss)
        if best_box == torch.tensor(0):
            object_loss = self.mse(prediction[..., 20:21]*exists_box,target[...,20:21]*exists_box)
        elif best_box == torch.tensor(1): 
            object_loss = self.mse(prediction[..., 25:26]*exists_box,target[...,20:21]*exists_box)

        no_object_loss_b1 = self.mse(prediction[...,20:21]*(1-exists_box), target[...,20:21]*(1-exists_box))
        no_object_loss_b2 = self.mse(prediction[..., 25:26]*(1-exists_box),target[...,20:21]*(1-exists_box))
        no_object_loss = self.noobj * (no_object_loss_b1 + no_object_loss_b2)   
        class_loss = self.mse(prediction[...,:20]*exists_box, target[...,:20]*exists_box)
            
        total_loss = box_loss + object_loss + no_object_loss + class_loss
        
        return total_loss
        