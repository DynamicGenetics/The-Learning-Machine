"""
Specify Data sources used to proxy access to available Torch Datasets
"""

import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, ConcatDataset
from .fer import FER
from .ferplus import FERPlus
from typing import Callable, Sequence, Union, Set
from PIL.Image import Image as PILImage
from random import sample
from hashlib import sha256
from os import path
from pathlib import Path

SECRET_SPICE = "supersecrectspiceonthebackend"


@dataclass
class Sample:
    index: int
    emotion: int
    image: PILImage

    def emotion_label(self, classes):
        return classes[self.emotion]

    @property
    def uuid(self) -> str:
        encode_s = f"{SECRET_SPICE}_{self.emotion}"
        encode_b = sha256(bytes(encode_s, encoding="utf8")).hexdigest()
        ref = hex(self.index)[2:]  # getting rid of 0x
        uuid = f"{encode_b}_{ref}"
        return uuid

    def __hash__(self) -> int:
        return int(self.uuid, base=16)

    def __iter__(self):
        return iter((self,))

    @staticmethod
    def retrieve_index(uuid: str) -> int:
        try:
            _, ref = uuid.split("_")
            return int(ref, base=16)
        except ValueError:
            return -1


DatasetLoadFn = Callable[[], Dataset]


def load_fer_dataset() -> Dataset:
    default_root_folder = path.dirname(path.abspath(__file__))
    fer_train = FER(root=default_root_folder, download=True, split="train")
    fer_valid = FER(root=default_root_folder, download=True, split="validation")
    fer_test = FER(root=default_root_folder, download=True, split="test")
    return ConcatDataset([fer_train, fer_valid, fer_test])


def load_ferplus_dataset() -> Dataset:
    default_root_folder = path.dirname(path.abspath(__file__))
    fer_train = FERPlus(root=default_root_folder, download=True, split="train")
    fer_valid = FERPlus(
        root=default_root_folder,
        download=True,
        split="validation",
    )
    fer_test = FERPlus(root=default_root_folder, download=True, split="test")
    return ConcatDataset([fer_train, fer_valid, fer_test])


def only_fer_training() -> Dataset:
    default_root_folder = path.dirname(path.abspath(__file__))
    return FER(root=default_root_folder, download=True, split="train")


def only_fer_validation() -> Dataset:
    default_root_folder = path.dirname(path.abspath(__file__))
    return FER(root=default_root_folder, download=True, split="validation")


def only_ferplus_validation() -> Dataset:
    default_root_folder = path.dirname(path.abspath(__file__))
    return FERPlus(root=default_root_folder, download=True, split="validation")


class DataSource:

    BLACKLIST_SAMPLES = Path("indices_blacklist.txt")
    RETURNED_SAMPLES = Path("indices_sampled.txt")

    def __init__(
        self,
        dataset_load_fn: DatasetLoadFn = load_fer_dataset,
        target_emotions: Sequence[str] = FER.classes,
    ) -> None:
        self._dataset = None  # Instantiated once via property
        self._ds_load_fn = dataset_load_fn
        self._items_sampled = self._init_list(
            self.RETURNED_SAMPLES
        )  # implementing memory-map
        self._blacklist = self._init_list(
            self.BLACKLIST_SAMPLES
        )  # Set of indices to exclude *ever*
        self._emotions = target_emotions

    def _init_list(self, samples_list_filepath: Path) -> Set:
        if samples_list_filepath.exists():
            return self._load_from(samples_list_filepath)
        return set()

    @staticmethod
    def _load_from(samples_list_filepath: Path) -> Set:
        with open(samples_list_filepath) as f:
            indices = f.read().strip().split(",")
            indices = filter(lambda idx: len(idx.strip()) > 0, indices)
            indices = map(int, indices)
        indices = set(indices)
        return indices

    @staticmethod
    def _serialise(indices: Set, target_list_filepath: Path) -> None:
        with open(target_list_filepath, "w") as f:
            line = ",".join(map(str, indices))
            f.write(f"{line}\n")

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            self._dataset = self._ds_load_fn()
        return self._dataset

    @property
    def emotions(self) -> Sequence[str]:
        return self._emotions

    def emotion_index(self, emotion_label: str) -> int:
        try:
            idx = self._emotions.index(emotion_label)
        except ValueError:
            idx = -1
        return idx

    def __getitem__(self, index: Union[str, int]) -> Sample:
        try:
            index = int(index)
        except ValueError:
            index = Sample.retrieve_index(str(index))
        image, label = self.dataset[index]
        return Sample(index=index, emotion=label, image=image)

    def get_random_samples(self, k: int) -> Sequence[Sample]:
        samples = list()
        excluded = self._items_sampled.union(self._blacklist)
        pool = set(range(len(self.dataset))).difference(excluded)
        rnd_indices = sample(pool, k=k)
        for sample_idx in rnd_indices:
            samples.append(self[sample_idx])
            self._items_sampled.add(sample_idx)
        return samples

    def discard_sample(self, index: Union[str, int]):
        try:
            index = int(index)
        except ValueError:
            index = Sample.retrieve_index(str(index))
        self._blacklist.add(index)

    def serialise_session(self) -> None:
        # Serialise Items Sampled
        # self._serialise(self._items_sampled, self.RETURNED_SAMPLES)
        # Serialise the MODEL WEIGHTS
        # TODO
        # Serialise Blacklist
        self._serialise(self._blacklist, self.BLACKLIST_SAMPLES)


class EvaluationFERDataSource(DataSource):

    # for more details, see notebooks/FER_AutoEncoder.ipynb
    SAMPLES_PER_EMOTION_INDEX = {
        "angry": [467, 1023, 1081, 3529, 2544, 560, 2148, 1340, 2236, 2711],
        "disgust": [911, 419, 614, 1372, 1486, 2482, 2650, 2495, 3488, 2221],
        "fear": [358, 2551, 515, 440, 1325, 994, 3040, 2498, 709, 800],
        "happy": [840, 2138, 998, 3460, 2995, 225, 3124, 1501, 1617, 3227],
        "sad": [1051, 2974, 2116, 1145, 605, 1897, 1638, 1207, 285, 3050],
        "surprise": [161, 1034, 1018, 1333, 1095, 2095, 3035, 3215, 1692, 3386],
        "neutral": [2262, 1885, 1963, 3391, 1469, 1941, 2869, 2893, 2934, 1667],
    }

    def __init__(self) -> None:
        super().__init__(
            dataset_load_fn=only_fer_validation, target_emotions=FER.classes
        )
        self._evaluation_samples = None

    @property
    def evaluation_samples(self):
        if self._evaluation_samples is None:
            self._evaluation_samples = list()
            for emotion, indices in self.SAMPLES_PER_EMOTION_INDEX.items():
                if emotion == "neutral":
                    continue
                for sample_idx in indices:
                    self._evaluation_samples.append(self[sample_idx])
        return self._evaluation_samples

    @property
    def ground_truth(self):
        y_true = list()
        for emotion, indices in self.SAMPLES_PER_EMOTION_INDEX.items():
            if emotion == "neutral":
                continue
            y_true += [self.emotions.index(emotion)] * len(indices)
        return np.asarray(y_true)


class EvaluationFERPlusDataSource(DataSource):

    # for more details, see notebooks/FERPlus_AutoEncoder.ipynb
    SAMPLES_PER_EMOTION_INDEX = {
        "happy": [175, 1562, 1629, 531, 1905, 900, 2566],
        "surprise": [1873, 1137, 79, 1827, 1882, 2588, 823],
        "sad": [232, 1552, 2514, 983, 206, 2269, 901],
        "angry": [1242, 1038, 555, 1496, 559, 989, 513],
        "disgust": [2501, 1291, 281, 2012, 384, 2342, 467],
        "fear": [1172, 2196, 2528, 1287, 704, 237, 1049],
    }

    def __init__(self) -> None:
        super().__init__(
            dataset_load_fn=only_ferplus_validation, target_emotions=FERPlus.classes
        )
        self._evaluation_samples = None

    @property
    def evaluation_samples(self):
        if self._evaluation_samples is None:
            self._evaluation_samples = list()
            for emotion, indices in self.SAMPLES_PER_EMOTION_INDEX.items():
                for sample_idx in indices:
                    self._evaluation_samples.append(self[sample_idx])
        return self._evaluation_samples

    @property
    def ground_truth(self):
        y_true = list()
        for emotion, indices in self.SAMPLES_PER_EMOTION_INDEX.items():
            y_true += [self.emotions.index(emotion)] * len(indices)
        return np.asarray(y_true)


def load_fer_dataset_lazy() -> DataSource:
    """Instantiate a DataSource instance, proxying access to
    corresponding torch.Dataset.

    The DataSource lazy connects to the mapped dataset, holding
    reference to the database, and establishing actual connection
    only on the first instance access.
    """
    return DataSource()  # default load function


def load_ferplus_dataset_lazy() -> DataSource:
    """Instantiate a DataSource instance, proxying access to
    corresponding torch.Dataset for the FERPlus dataset.

    The DataSource lazy connects to the mapped dataset, holding
    reference to the database, and establishing actual connection
    only on the first instance access.
    """
    return DataSource(
        dataset_load_fn=load_ferplus_dataset, target_emotions=FERPlus.classes
    )  # default load function


def load_fer_training_lazy() -> DataSource:
    """Instantiate a DataSource instance, proxying access to
    corresponding torch.Dataset.

    The DataSource lazy connects to the mapped dataset, holding
    reference to the database, and establishing actual connection
    only on the first instance access.
    """
    return DataSource(dataset_load_fn=only_fer_training)  # default load function


def load_fer_validation_lazy() -> DataSource:
    """Instantiate a DataSource instance, proxying access to
    corresponding torch.Dataset.

    The DataSource lazy connects to the mapped dataset, holding
    reference to the database, and establishing actual connection
    only on the first instance access.
    """
    return DataSource(dataset_load_fn=only_fer_validation)  # default load function


# ======================
# EVALUATION DATA SOURCE
# ======================


def load_fer_evaluation_ds_lazy() -> DataSource:
    """Instantiate an EvaluationDataSource instance, proxying access to
    corresponding torch.Dataset.

    The EvaluationDataSource lazy connects to the mapped dataset, holding
    reference to the set of samples (per emotion) to be used to calculate
    metrics (i.e. performance) of the learning machine during training.
    """
    return EvaluationFERDataSource()


def load_ferplus_evaluation_ds_lazy() -> DataSource:
    """Instantiate an EvaluationDataSource instance, proxying access to
    corresponding torch.Dataset.

    The EvaluationDataSource lazy connects to the mapped dataset, holding
    reference to the set of samples (per emotion) to be used to calculate
    metrics (i.e. performance) of the learning machine during training.
    """
    return EvaluationFERPlusDataSource()
