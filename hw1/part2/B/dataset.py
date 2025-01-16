import os
import torch
import torchvision.io as tvio
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage

class ImageDataset(Dataset):
    def __init__(self, img_dir, dataset_img, dataset_mask, transform=None):
        self.img_dir = img_dir
        self.dataset_img = dataset_img
        self.dataset_mask = torch.Tensor(dataset_mask)
        self.transform = transform
        self.isprocessed = []

    def __len__(self):
        return len(self.dataset_img)

    def __getitem__(self, idx):
        img_name = self.dataset_img[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = tvio.read_image(img_path)
        mask = self.dataset_mask[idx]

        if self.transform:
            to_pil = ToPILImage()
            img = self.transform(to_pil(img))

        return img, mask