import torch
import torch.nn as nn
from torch.utils.data import DataLoader



class TrainTestSteps(nn.Module):
    def __init__(self):
        super().__init__()

    def train_step(self, model:nn.Module, dataloader:DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device:torch.device):
        model.train()

        train_loss =0
        
        for batch, (x,y) in enumerate(dataloader):
            x,y = 
            y_pred = model(x)

            loss_logits = loss_fn(y_pred,y)
            train_loss+= loss_logits.item()

