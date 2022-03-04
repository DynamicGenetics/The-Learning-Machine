import torch
from torch import nn
from collections import OrderedDict


class VGGFERNet(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 7):
        super(VGGFERNet, self).__init__()

        self.conv1 = self._block(in_channels, 64, name="conv1", n_conv_layers=1)
        self.conv2 = self._block(64, 128, name="conv2", n_conv_layers=1)
        self.conv3 = self._block(128, 256, name="conv3", n_conv_layers=1)

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("cl_fc_1", nn.Linear(1024 * 3 * 3, 1024)),
                    ("cl_relu_1", nn.ReLU(inplace=True)),
                    ("cl_drop_1", nn.Dropout(0.25)),
                    ("cl_fc_2", nn.Linear(1024, 256)),
                    ("cl_relu_2", nn.ReLU(inplace=True)),
                    ("cl_drop_2", nn.Dropout(0.25)),
                    ("output", nn.Linear(256, n_classes)),
                ]
            )
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    @staticmethod
    def _block(in_channels: int, features: int, name: str, n_conv_layers: int = 2):

        layers = list()
        for i in range(n_conv_layers):
            layers.append(
                (
                    name + f"conv{i+1}",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=(3, 3),
                        padding=1,
                        bias=False,
                    ),
                )
            )
            layers.append((name + f"relu{i+1}", nn.ReLU(inplace=True)))
            in_channels = features
        layers.append(
            (
                name + "pool",
                nn.MaxPool2d(kernel_size=(2, 2)),
            )
        )
        layers.append(
            (
                name + "dropout",
                nn.Dropout2d(p=0.25),
            )
        )

        return nn.Sequential(OrderedDict(layers))
