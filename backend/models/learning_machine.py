"""
LearningMachine Abstract base class definition
"""

from typing import Any, NoReturn, Sequence, Tuple
from nptyping import NDArray
from numpy import float32 as float32
import torch
from abc import abstractmethod, ABC
from torch import nn, optim, Tensor
from torch.utils.data.dataloader import default_collate
from pathlib import Path
import os
from torchvision.transforms import ToTensor
from torchvision.datasets.utils import download_url
from datasets import Sample
from typing import Callable, Union, Dict, Optional
from PIL.Image import Image as PILImage

# Types
ModelOutput = Union[Tensor, Tuple[Tensor, Tensor]]
Prediction = NDArray[
    (
        Any,
        7,
    ),
    float32,
]
TransformerType = Callable[[Union[Sequence[Callable], PILImage, Tensor]], Tensor]
StateDictType = Union[Dict[str, Tensor], Dict[str, Tensor]]

BASE_FOLDER = Path(os.path.dirname(os.path.abspath(__file__)))
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LearningMachine(ABC):
    """ """

    CHECKPOINTS_FOLDER = BASE_FOLDER / "weights"

    def __init__(self) -> None:
        self._model = None
        self._weights = None
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
