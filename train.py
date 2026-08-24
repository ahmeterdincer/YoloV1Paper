import torch
import torch.optim
from torchvision import transforms
from config import S,B,C,BATCH_SIZE,LEARNING_RATE,EPOCHS,DEVICE,VOC_ROOT, VOC_CLASSES_LENGTH
from model import YOLOv1
from loss import YOLOLoss
from engine import TrainTestSteps
import dataset
from dataset import Dataset
from torch.utils.data import DataLoader


train_transforms, test_transforms = dataset.feature_extracture()
train_raw, test_raw = dataset.data_load(voc_root=VOC_ROOT)

train_dataset = Dataset(train_raw, S, B, C, transform=train_transforms)
test_dataset  = Dataset(test_raw,  S, B, C, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

model = YOLOv1(in_channels=C, S=S, B=B,C=VOC_CLASSES_LENGTH).to(device=DEVICE)
loss_val = YOLOLoss(S=S, B=B, C=VOC_CLASSES_LENGTH)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

trainer = TrainTestSteps()

for i in range(EPOCHS):
    train_loss= trainer.train_step(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_val)
    test_loss = trainer.test_step(model=model, dataloader=test_loader, loss_fn=loss_val)
    print(f"Epoch {i+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    torch.save(model.state_dict(), "yolov1.pth")
    
