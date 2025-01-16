import os
import sys
import csv
import torch
import numpy as np
import torch.nn as nn
import torchvision.io as tvio
from torchvision.models import resnet50
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, img_dir, dataset, transform=None):
        self.img_dir = img_dir
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = self.dataset[idx]
        self.img_name = img_name
        img_path = os.path.join(self.img_dir, img_name)
        img = tvio.read_image(img_path)

        if self.transform:
            to_pil = ToPILImage()
            img = self.transform(to_pil(img))

        if len(img_name.split("_")) > 1:
            label = img_name.split("_")[0]
            label = torch.tensor(int(label), dtype=torch.float32)
            return img, label
        else:
            return img

test_csv = sys.argv[1]
img_dir_val = sys.argv[2]
test_pred_csv = sys.argv[3]

output_dir = os.path.dirname(test_pred_csv)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory {output_dir} created.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_val = transforms.Compose([
    transforms.Resize(128, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_dataset = [file for file in os.listdir(img_dir_val) if file.endswith(".jpg")]
val_data = ImageDataset(img_dir_val, val_dataset, transform=transform_val)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

model = resnet50(weights=None).to(device)
model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 65))
model.load_state_dict(torch.load("inference_model/p1_model.pt"))
model = model.to(device)

model.eval()
predict_list = []
correct_test, total_test = 0, 0
with torch.no_grad():
    for i, (images, _) in enumerate(val_loader):
        test_images = images.to(torch.float32).to(device)
        outputs = model(test_images)
        predict = torch.max(outputs, 1)[1]
        predict_list.extend(predict.flatten().detach().tolist())

np_img_name = np.array(val_dataset, dtype=str)
np_predict = np.array(predict_list, dtype=np.uint8)

test_img_name = []
with open(test_csv, "r", newline="") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        if len(row) > 1:
            test_img_name.append(row[1])

with open(test_pred_csv, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("id", "filename", "label"))
    id = 0
    for img_name, predict in zip(np_img_name, np_predict):
        if img_name in test_img_name:
            writer.writerow([id, img_name, predict])
            id += 1

print("Finish p1_inference.py")