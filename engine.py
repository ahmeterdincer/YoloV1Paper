import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm


class TrainTestSteps(nn.Module):
    """Modelin eğitim ve test adımlarını yöneten sınıf."""
    def __init__(self):
        super().__init__()

    def train_step(self, model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device):
        """Bir epoch boyunca modeli eğitir ve ortalama kayıp (mean loss) değerini döndürür."""
        model.train()

        train_loss = 0
        mean_loss = []
        loop = tqdm(dataloader, leave=True)
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_loss.append(loss.item())
            loop.set_postfix(loss=loss.item())

        return sum(mean_loss) / len(mean_loss)

    def test_step(self, model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device):
        """Modeli test/doğrulama veri seti üzerinde değerlendirir ve ortalama test kaybını döndürür."""
        model.eval()
        test_loss=0
        
        with torch.inference_mode():
            for batch,(x,y) in enumerate(dataloader):
                x,y = x.to(device), y.to(device)
                test_pred = model(x)

                loss = loss_fn(test_pred,y)
                test_loss += loss.item()
            
            test_loss = test_loss/len(dataloader) 
            return test_loss