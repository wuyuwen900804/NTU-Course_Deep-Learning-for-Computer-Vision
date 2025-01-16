import os
import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torchvision.transforms as transforms

from train import fit_model
from dataset import ImageDataset

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
set_seed(0)

lr, batch_size, epochs = 1e-3, 32, 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_dir_train = "hw1_data/p1_data/office/train"
img_dir_val = "hw1_data/p1_data/office/val"

transform_train = transforms.Compose([
    transforms.Resize(128, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(128),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(128, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = [file for file in os.listdir(img_dir_train) if file.endswith(".jpg")]
val_dataset = [file for file in os.listdir(img_dir_val) if file.endswith(".jpg")]
train_data = ImageDataset(img_dir_train, train_dataset, transform=transform_train)
val_data = ImageDataset(img_dir_val, val_dataset, transform=transform_val)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=4)

model = resnet50(weights=None)
model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 65))
model = model.to(device)
parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(parameters, lr=lr)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    training_loss, training_acc, val_loss, val_acc = fit_model(
        model, criterion, optimizer, epochs, train_loader, val_loader
    )