import torch.nn as nn
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