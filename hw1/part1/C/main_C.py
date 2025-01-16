import os
import torch
import random
import numpy as np
import torch.nn as nn
from byol_pytorch import BYOL
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torchvision.transforms as transforms

from dataset import ImageDataset
from train import SSL_fit_model, FT_fit_model

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

SSL_lr, SSL_batch_size, SSL_epochs = 1e-2, 32, 50
FT_lr, FT_batch_size, FT_epochs = 1e-3, 32, 100
device = "cuda" if torch.cuda.is_available() else "cpu"

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

SSL_img_dir_train = "hw1_data/p1_data/mini/train"
SSL_train_dataset = [file for file in os.listdir(SSL_img_dir_train) if file.endswith(".jpg")]
SSL_train_data = ImageDataset(SSL_img_dir_train, SSL_train_dataset, transform=transform_train)
SSL_train_loader = DataLoader(SSL_train_data, batch_size=SSL_batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)

FT_img_dir_train = "hw1_data/p1_data/office/train"
FT_img_dir_val = "hw1_data/p1_data/office/val"
FT_train_dataset = [file for file in os.listdir(FT_img_dir_train) if file.endswith(".jpg")]
FT_val_dataset = [file for file in os.listdir(FT_img_dir_val) if file.endswith(".jpg")]
FT_train_data = ImageDataset(FT_img_dir_train, FT_train_dataset, transform=transform_train)
FT_val_data = ImageDataset(FT_img_dir_val, FT_val_dataset, transform=transform_val)
FT_train_loader = DataLoader(FT_train_data, batch_size=FT_batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
FT_val_loader = DataLoader(FT_val_data, batch_size=FT_batch_size,
                           shuffle=True, num_workers=4, pin_memory=True)


if __name__ == "__main__":
    SSL = False
    if SSL == True:
        SSL_model = resnet50(weights=None).to(device)
        # SSL_model.load_state_dict(torch.load("part1/PretrainBYOL.pt"))
        SSL_learner = BYOL(SSL_model, image_size=128, hidden_layer="avgpool")
        SSL_optimizer = torch.optim.Adam(SSL_learner.parameters(), lr=SSL_lr)
        SSL_fit_model(SSL_learner, SSL_model, SSL_optimizer, SSL_epochs, SSL_train_loader)

    FT_model = resnet50(weights=None).to(device)
    FT_model.load_state_dict(torch.load("part1/PretrainBYOL.pt"))
    FT_model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 65))
    FT_model = FT_model.to(device)
    FT_parameters = filter(lambda p: p.requires_grad, FT_model.parameters())
    FT_optimizer = torch.optim.Adam(FT_parameters, lr=FT_lr)
    FT_criterion = nn.CrossEntropyLoss()
    FT_training_loss, FT_training_acc, FT_val_loss, FT_val_acc = FT_fit_model(
        FT_model, FT_criterion, FT_optimizer, FT_epochs, FT_train_loader, FT_val_loader
    )