import os
import sys
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import imageio.v2 as imageio
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)

class Deeplabv3_MobilenetV3_Model(nn.Module):
    def __init__(self):
        super(Deeplabv3_MobilenetV3_Model, self).__init__()
        self.model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        self.model.classifier[4] = nn.Conv2d(256, 7, kernel_size=1)

    def forward(self, x):
        output = self.model(x)
        return output["out"]


class ImageDataset(Dataset):
    def __init__(self, file_path, img_transform=None):
        self.path = file_path
        self.data = []
        self.imgfile = sorted([img for img in os.listdir(self.path) if img.endswith("sat.jpg")])

        for img in self.imgfile:
            self.data.append(Image.open(os.path.join(self.path, img)).copy())
        self.transform = img_transform

    def __len__(self):
        return len(self.imgfile)
    
    def __getitem__(self, idx):
        data = self.data[idx]

        if self.transform:
            data = self.transform(data)

        return data, self.imgfile[idx]


def mean_iou_score(pred, labels):
    """
    Compute mean IoU score over 6 classes
    """
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6

    return mean_iou

def pred2image(batch_preds, batch_names, out_path):
    for pred, name in zip(batch_preds, batch_names):
        pred = pred.detach().cpu().numpy()
        pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
        color_map = {
            0: [0, 255, 255],  # Cyan (Urban land)
            1: [255, 255, 0],  # Yellow (Agriculture land)
            2: [255, 0, 255],  # Purple (Rangeland)
            3: [0, 255, 0],    # Green (Forest land)
            4: [0, 0, 255],    # Blue (Water)
            5: [255, 255, 255],# White (Barren land)
            6: [0, 0, 0],      # Black (Unknown)
        }
        for label, color in color_map.items():
            pred_img[pred == label] = color
        imageio.imwrite(os.path.join(out_path, name.replace("sat.jpg", "mask.png")), pred_img)

img_dir_valid = sys.argv[1]
output_folder = sys.argv[2]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Directory {output_folder} created.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_valid = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_data = ImageDataset(img_dir_valid, img_transform=transform_valid)
valid_loader = DataLoader(dataset=valid_data, batch_size=4)

model = Deeplabv3_MobilenetV3_Model()
model = model.to(device)
model.load_state_dict(torch.load("inference_model/p2_model.pt"))

val_loss, val_pred_list, val_mask_list = [], [], []

model.eval()
with torch.no_grad():
    for i, (imgs, filenames) in enumerate(valid_loader):
        imgs = imgs.to(device)
        output = model(imgs)
        pred = output.cpu().argmax(dim=1)
        val_pred_list.append(pred)
        pred2image(pred, filenames, output_folder)

print("Finish p2_inference.py")