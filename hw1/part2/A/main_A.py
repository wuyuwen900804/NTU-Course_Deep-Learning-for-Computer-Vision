import os
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from train import fit_model
from model import VGG16_FCN32s
from dataset import ImageDataset
from mean_iou_evaluate import read_masks

lr, batch_size, epochs = 5e-4, 4, 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir_train = "hw1_data/p2_data/train"
img_dir_valid = "hw1_data/p2_data/validation"

transform_train = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_valid = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = [imgfile for imgfile in os.listdir(img_dir_train) if imgfile.endswith(".jpg")]
valid_dataset = [imgfile for imgfile in os.listdir(img_dir_valid) if imgfile.endswith(".jpg")]
train_dataset.sort()
valid_dataset.sort()

train_data = ImageDataset(img_dir_train, train_dataset, read_masks(img_dir_train), transform_train)
valid_data = ImageDataset(img_dir_valid, valid_dataset, read_masks(img_dir_valid), transform_valid)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)

model = VGG16_FCN32s(7)
model = model.to(device)
parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(parameters, lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=6)

if __name__ == '__main__':
    # summary(model, (3, 512, 512))
    train_loss, train_iou, valid_loss, valid_iou = fit_model(
        model, criterion, optimizer, epochs, train_loader, valid_loader
    )