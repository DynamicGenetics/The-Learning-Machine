"""
LearningMachine Abstract base class definition
"""

import os
from pathlib import Path

from numpy import float32 as float32
from numpy import log as np_log
from abc import abstractmethod, ABC

import torch
from torch import nn, optim, Tensor
from torch.utils.data.dataloader import default_collate

from torchvision.transforms import ToTensor, Compose, Lambda
from torchvision.datasets.utils import download_url

# Typing
from typing import Any, NoReturn, Sequence, Tuple, Optional
from typing import Callable, Union, Dict, Optional
from nptyping import NDArray
from PIL.Image import Image as PILImage

# Models
from .unet import Unet
from .vgg_fer import VGGFERNet
from .vgg import VGG13Net

from datasets import Sample

ModelOutput = Union[Tensor, Tuple[Tensor, Tensor]]
Prediction = NDArray[
    (
        Any,
        7,
    ),
    float32,
]
TransformerType = Callable[[Union[Sequence[Callable], PILImage, Tensor]], Tensor]
StateDictType = (
    "OrderedDict[str, Tensor]"  # Union[Dict[str, Tensor], Dict[str, Tensor]]
)

BASE_FOLDER = Path(os.path.dirname(os.path.abspath(__file__)))
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LearningMachine(ABC):
    """ """

    CHECKPOINTS_FOLDER = BASE_FOLDER / "weights"

    def __init__(self, pretrained: bool = False) -> None:
        self._model = None
        self._weights = None
        self._pretrained = pretrained
        self._transformer = self._set_transformer()
        self._criterion = self._init_criterion()
        self._optimiser = self._init_optimiser()

        os.makedirs(self.CHECKPOINTS_FOLDER, exist_ok=True)

    @staticmethod
    def _set_transformer() -> TransformerType:
        """Default transformer: always convert an image to a torch tensor!"""
        return ToTensor()

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._model = self._load_model()
            # Move model instance to the target memory location
            self._model = self._model.to(TORCH_DEVICE)
        return self._model

    @property
    @abstractmethod
    def checkpoint(self) -> Path:
        pass

    @property
    @abstractmethod
    def weights_urls(self) -> Tuple[str, str]:
        pass

    @property
    def optimiser(self) -> optim.Optimizer:
        if self._optimiser is None:
            self._optimiser = self._init_optimiser()
        return self._optimiser

    @property
    def criterion(self) -> nn.Module:
        if self._criterion is None:
            self._criterion = self._init_criterion()
        return self._criterion

    @property
    def weights(self) -> StateDictType:
        if self._weights is None:
            print(f"[INFO]: loading {self.checkpoint}")
            if not self.checkpoint.exists():
                self._download_weights()
            self._weights = torch.load(self.checkpoint, map_location=TORCH_DEVICE)
        return self._weights

    @abstractmethod
    def _load_model(self) -> nn.Module:
        raise NotImplementedError("You should not instantiate a Model explicitly.")

    @abstractmethod
    def _init_optimiser(self) -> optim.Optimizer:
        pass

    @abstractmethod
    def _init_criterion(self) -> nn.Module:
        pass

    @property
    def is_pretrained(self) -> bool:
        return self._pretrained

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def _download_weights(self) -> NoReturn:
        # download weights files

        # MONKEY Patch torchvision utils
        from torchvision.datasets import utils

        utils._get_redirect_url = lambda url, max_hops: url

        url, md5 = self.weights_urls
        filename = url.rpartition("/")[-1].split("?")[0]
        download_url(url, root=self.CHECKPOINTS_FOLDER, filename=filename, md5=md5)

    def _calculate_loss(
        self,
        labels: Tensor,
        model_output: ModelOutput,
        input_batch: Optional[Tensor] = None,
    ) -> Tensor:
        loss = self.criterion(model_output, labels)
        return loss

    def _model_call(self, batch: Sequence[Sample]) -> Tensor:
        return self.model(batch)

    def transform(self, sample: Sample) -> Tensor:
        return self._transformer(sample.image)

    def predict(
        self, samples: Union[Sample, Sequence[Sample]], as_proba: bool = True
    ) -> Prediction:
        """

        Parameters
        ----------
        samples : Sequence[Sample]
            The Sequence of sample instances to generate predictions for
        as_proba : bool (default True)
            If True, returns predictions as probabilities. Otherwise, just
            logits will be returned

        Returns
        -------
            Numpy Array of shape (n_samples x  n_emotions)
        """
        # transform samples into a batch of torch Tensors
        batch = default_collate(list(map(self.transform, iter(samples))))
        with torch.no_grad():
            self.model.eval()
            batch = batch.to(TORCH_DEVICE)
            outputs = self._model_call(batch)
            outputs = self._get_model_emotion_predictions(outputs)
            if not as_proba:
                return outputs  # return logits
            outputs_min = outputs.min(axis=1, keepdims=True)[0]
            outputs_max = outputs.max(axis=1, keepdims=True)[0]
            probabilities = (outputs - outputs_min) / (outputs_max - outputs_min)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return probabilities

    @staticmethod
    def _get_model_emotion_predictions(model_output: ModelOutput) -> Prediction:
        return model_output.detach().cpu()

    def fit(self, samples: Sequence[Sample]) -> NoReturn:
        """ """
        # convert the input sequence of Samples into a batch
        # of torch Tensor
        batch = default_collate(list(map(self.transform, iter(samples))))
        labels = default_collate([s.emotion for s in iter(samples)])
        with torch.set_grad_enabled(True):
            self.model.train()
            # zero the gradient
            self.optimiser.zero_grad()
            # forward pass
            batch = batch.to(TORCH_DEVICE)
            labels = labels.to(TORCH_DEVICE)
            outputs = self._model_call(batch)
            loss = self._calculate_loss(
                labels=labels, model_output=outputs, input_batch=batch
            )
            # backward + optimize
            loss.backward()
            self.optimiser.step()

    def __call__(self, samples: Sequence[Sample]) -> Prediction:
        return self.predict(samples=samples)


class UNetMachine(LearningMachine):
    """Unet-based Learning Machine"""

    def __init__(self, loss_reco_weight: float = 0.3, pretrained: bool = False):
        super(UNetMachine, self).__init__(pretrained=pretrained)
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
        if self._pretrained:
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

    @property
    def name(self) -> str:
        return "UNet"

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


class VGGMachine(LearningMachine):
    """VGG-based Learning Machine"""

    def __init__(self, pretrained: bool = False) -> None:
        super(VGGMachine, self).__init__(pretrained=pretrained)

    @staticmethod
    def _set_transformer() -> TransformerType:
        def _convert_rgb(img: PILImage) -> PILImage:
            return img.convert("RGB")

        return Compose([Lambda(_convert_rgb), ToTensor()])

    @property
    def checkpoint(self) -> Path:
        return self.CHECKPOINTS_FOLDER / "vgg_learning_machine_overfitting.pt"

    @property
    def weights_urls(self) -> Tuple[str, str]:
        return (
            "https://www.dropbox.com/s/2q68kitijwona2l/vgg_learning_machine_overfitting.pt?dl=1",
            "e76c032150d9762e94a6b94b3d5c2b9d",
        )

    def _init_optimiser(self) -> optim.Optimizer:
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def _init_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _load_model(self) -> nn.Module:
        model = VGG13Net(pretrained=False, freeze=False)
        model.load_state_dict(self.weights)
        return model


class VGGFERMachine(LearningMachine):
    """VGG-based Learning Machine"""

    def __init__(self, pretrained: bool = False) -> None:
        super(VGGFERMachine, self).__init__(pretrained=pretrained)

    @staticmethod
    def _set_transformer() -> TransformerType:
        def _convert_rgb(img: PILImage) -> PILImage:
            return img.convert("RGB")

        return Compose([Lambda(_convert_rgb), ToTensor()])

    @property
    def checkpoint(self) -> Path:
        return self.CHECKPOINTS_FOLDER / "TBD"

    @property
    def weights_urls(self) -> Tuple[str, str]:
        return (
            "TBD",
            "TBD",
        )

    def _init_optimiser(self) -> optim.Optimizer:
        momentum = -32 / np_log(0.9)  # -minibatch_size/log(0.9)
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=momentum)

    def _init_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _load_model(self) -> nn.Module:
        model = VGGFERNet(in_channels=3)
        # if self._pretrained:
        #     model.load_state_dict(self.weights)
        return model

    @property
    def name(self) -> str:
        return "VGG"
