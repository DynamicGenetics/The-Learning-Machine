from pathlib import Path

import torch
from torch import nn, optim, Tensor
from collections import OrderedDict
from .learning_machine import LearningMachine
from .learning_machine import Prediction, ModelOutput
from datasets import Sample
from typing import Sequence, Tuple, Optional, Union


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, filters=16, n_classes=7):
        super(Unet, self).__init__()

        self.encoder1 = self._block(in_channels, filters, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(filters, filters * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(filters * 2, filters * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(filters * 4, filters * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(filters * 8, filters * 16, name="bottleneck")

        self.fc = nn.Linear(256 * 3 * 3, n_classes)

        self.upconv4 = nn.ConvTranspose2d(
            filters * 16, filters * 8, kernel_size=2, stride=2
        )

        self.decoder4 = self._block((filters * 8) * 2, filters * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(
            filters * 8, filters * 4, kernel_size=2, stride=2
        )

        self.decoder3 = self._block((filters * 4) * 2, filters * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose2d(
            filters * 4, filters * 2, kernel_size=2, stride=2
        )

        self.decoder2 = self._block((filters * 2) * 2, filters * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2)

        self.decoder1 = self._block(filters * 2, filters, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=filters, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        """"""
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        encoding = bottleneck.view(-1, 256 * 3 * 3)
        fe = self.fc(encoding)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        reco = torch.clamp(self.conv(dec1), min=0, max=1)
        return reco, fe

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class UNetMachine(LearningMachine):
    """Unet-based Learning Machine"""

    def __init__(self, loss_reco_weight: float = 0.3):
        super(UNetMachine, self).__init__()
        self.loss_reco_coeff = loss_reco_weight
        self._reconstruction_criterion, self._prediction_criterion = self._criterion

    @property
    def checkpoint(self) -> Path:
        return self.CHECKPOINTS_FOLDER / "unet_learning_machine_nodecay_aug.pt"

    @property
    def weights_urls(self) -> Tuple[str, str]:
        return (
            "https://www.dropbox.com/s/nctn4x49t2xf6sq/"
            + "unet_learning_machine_nodecay_aug.pt?dl=1",
            "dbbd8866c5c6c7497feae735dd1513ce",
        )

    def _load_model(self):
        model = Unet()
        model.load_state_dict(self.weights)
        return model

    def _init_optimiser(self):
        return optim.Adam(self.model.parameters(), lr=0.0001)

    def _init_criterion(self) -> Tuple[nn.Module, nn.Module]:
        reco_criterion = nn.MSELoss(reduction="mean")
        pred_criterion = nn.CrossEntropyLoss()
        return reco_criterion, pred_criterion

    @property
    def criterion(self) -> Tuple[nn.Module, nn.Module]:
        if self._criterion is None:
            (
                self._reconstruction_criterion,
                self._prediction_criterion,
            ) = self._init_criterion()
            self._criterion = (
                self._reconstruction_criterion,
                self._prediction_criterion,
            )
        return self._criterion

    @property
    def reconstruction_criterion(self):
        return self._reconstruction_criterion

    @property
    def prediction_criterion(self):
        return self._prediction_criterion

    def _model_call(self, batch: Sequence[Sample]) -> Tuple[Tensor, Tensor]:
        return self.model(batch)

    def _calculate_loss(
        self,
        labels: Tensor,
        model_output: ModelOutput,
        input_batch: Optional[Tensor] = None,
    ) -> Tensor:
        reco_images, emotions_logits = model_output
        loss_reco = self.reconstruction_criterion(reco_images, input_batch)
        loss_pred = self.prediction_criterion(emotions_logits, labels)
        loss = (self.loss_reco_coeff * loss_reco) + (
            1.0 - self.loss_reco_coeff
        ) * loss_pred
        return loss

    @staticmethod
    def _get_model_emotion_predictions(model_output: ModelOutput) -> Prediction:
        reco_images, emotions_logits = model_output
        return emotions_logits.detach().cpu()
