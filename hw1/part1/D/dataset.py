import os
import torch
import torchvision.io as tvio
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage

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
        label = img_name.split("_")[0]
        label = torch.tensor(int(label), dtype=torch.float32)

        if self.transform:
            to_pil = ToPILImage()
            img = self.transform(to_pil(img))

        return img, label