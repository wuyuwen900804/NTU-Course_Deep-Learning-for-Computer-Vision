import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from mean_iou_evaluate import mean_iou_score
import torchvision.transforms as transforms

from model import VGG16_FCN32s
from dataset import ImageDataset
from mean_iou_evaluate import read_masks

img_dir_valid = "hw1_data/p2_data/validation"

transform_valid = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_dataset = [imgfile for imgfile in os.listdir(img_dir_valid) if imgfile.endswith(".jpg")]
valid_dataset.sort()
valid_data = ImageDataset(img_dir_valid, valid_dataset, read_masks(img_dir_valid), transform_valid)
valid_loader = DataLoader(dataset=valid_data, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

model = VGG16_FCN32s(7)
model.load_state_dict(torch.load("part2/A_best_iou.pt"))
model = model.to(device)

# validation phase
val_prediction_list, val_masks_list = [], []
model.eval()
with torch.no_grad():
    for i, (images, masks) in enumerate(valid_loader):
        test_images = images.to(torch.float32).to(device)
        test_masks = masks.to(torch.float32).to(device)

        outputs = model(test_images)

        prediction = torch.max(outputs, 1)[1]
        val_prediction_list.extend(prediction.cpu().numpy().flatten())
        val_masks_list.extend(masks.cpu().numpy().flatten())

val_iou = mean_iou_score(np.array(val_prediction_list), np.array(val_masks_list))