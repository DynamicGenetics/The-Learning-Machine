"""VGG13 based model for Learning Machine """

import torch
from torch import nn
from torchvision.models import vgg13
from typing import Any


class VGG13Net(nn.Module):
    """Custom VGG13 model architecture"""

    def __init__(
        self, freeze: bool = False, pretrained: bool = False, n_classes: int = 7
    ):
        super(VGG13Net, self).__init__()
        vgg13_architecture = vgg13(pretrained=pretrained, progress=pretrained)
        self.features = vgg13_architecture.features
        self.avgpool = vgg13_architecture.avgpool
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def __call__(self, *args, **kwargs) -> Any:
        return super().__call__(*args, **kwargs)
