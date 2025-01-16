import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG16_FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(VGG16_FCN32s, self).__init__()

        vgg = models.vgg16(weights='DEFAULT')
        
        self.features = vgg.features

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(1024, num_classes, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.upscore = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            kernel_size=44,
            stride=52,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.upscore(x)
        return x