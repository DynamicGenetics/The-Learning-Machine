"""
This module provides access to the FER+ (Facial Emotion Recognition Plus)
as encapsulated as a `torchvision.datasets.VisionDataset` class.

Notes
-----
The FER+ annotations provide a set of new labels for the standard FER dataset.
In FER+, each image has been labeled by 10 crowd-sourced taggers, which provide
better quality ground truth for each image/emotion than the original FER labels.

Having 10 taggers for each image enables researchers to estimate an emotion
probability distribution per face.
This allows constructing algorithms that produce statistical distributions or
multi-label outputs instead of the conventional single-label output,
as described in [1]_

The new label file is named `fer2013new.csv` and contains the same number of rows
as the original `fer2013.csv` label file with the same order,
so that you infer which emotion tag belongs to which image.

The format of the CSV file is as follows:
```
usage, neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF
```
Columns "usage" is the same as the original FER label to differentiate between
_Training_, _Public test_, and _Private test_ (validation) sets.

The other columns are the **vote count** for each emotion with the addition of
`unknown` and `NF` (i.e. _Not a Face_).


References
-----------
.. [1]  Emad Barsoum and Cha Zhang and Cristian Canton Ferrer and Zhengyou Zhang.
   "Training Deep Networks for Facial Expression Recognition with Crowd-Sourced
   Label Distribution". ICMI '16: Proceedings of the 18th ACM International
   Conference on Multimodal Interaction, October 2016, Pages 279–283
   https://doi.org/10.1145/2993148.2993165
"""
from typing import List, Union
from typing import Any, Optional, Callable
from pathlib import Path

import pandas as pd
import numpy as np
import torch as th

from .fer import FER, Partition


class FERPlus(FER):

    NEW_LABELS_DATA_FILE = "fer2013new.csv"
    RAW_DATA_FOLDER = "fer2013new"

    resources = [
        (
            "https://www.dropbox.com/s/659oxqg0osbozmj/fer2013new.tar.gz?dl=1",
            "338fafaa116322c4d7ecd24a65d014bf",
        )
    ]

    # NOTE: These are NOT classes original names (as in FER+) - aligned with FER! - and
    # they are not even in the original FER+ order: last two classes have been swapped!
    # However, pretrained model (i.e. VGGFERNet) has been trained on 8 classes
    # namely, include_nc=False). Therefore, model predictions are generated considering
    # a class mapping as ordered below.
    classes = [
        "neutral",
        "happy",
        "surprise",
        "sad",
        "angry",
        "disgust",
        "fear",
        "contempt",
        "not-human-face",
        "unknown",
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
        transform: Optional[Callable[[Any], Any]] = None,
        include_nf_class: bool = False,
    ):
        self._include_nf = include_nf_class
        super(FERPlus, self).__init__(
            root, split=split, download=download, transform=transform
        )

    @property
    def processed_folder(self):
        return (
            Path(self.root)
            / f"{self.__class__.__name__}{'_with_nc' if self._include_nf else ''}"
            / "processed"
        )

    @property
    def raw_folder(self):
        return (
            Path(self.root)
            / f"{self.__class__.__name__}{'_with_nc' if self._include_nf else ''}"
            / "raw"
        )

    @staticmethod
    def majority_count(entries: List[Union[str, int]]) -> Any:
        votes = entries[2:]
        all_votes = sum(votes)
        max_vote = max(votes)
        if max_vote <= 0.5 * all_votes:
            return len(votes) - 2  # UKNOWN emotion
        return np.argmax(votes)

    def _process_partitions(self):
        # data files path
        raw_data_filepath = self.raw_folder / self.RAW_DATA_FOLDER / self.RAW_DATA_FILE
        new_labels_data_filepath = (
            self.raw_folder / self.RAW_DATA_FOLDER / self.NEW_LABELS_DATA_FILE
        )
        # data frames
        raw_df = pd.read_csv(raw_data_filepath)
        fer_plus_df = pd.read_csv(new_labels_data_filepath, header=0)

        # count majority vote
        fer_plus_df["majority"] = fer_plus_df.apply(self.majority_count, axis=1)

        # set partitions
        raw_df["data_partition"] = raw_df.Usage.apply(self._set_partition)
        fer_plus_df["data_partition"] = fer_plus_df.Usage.apply(self._set_partition)

        for partition in Partition:
            dataset = raw_df[raw_df["data_partition"] == partition.value]
            fer_plus_ds = fer_plus_df[fer_plus_df["data_partition"] == partition.value]
            if not self._include_nf:
                valid = fer_plus_ds[fer_plus_ds.majority <= 7]  # all but UNKNWN and NF
            else:
                valid = fer_plus_ds[fer_plus_ds.majority != 8]  # all but UNKNWN
            images = dataset.loc[valid.index]
            images = self._images_as_torch_tensors(images)
            labels = self._labels_as_torch_tensors(valid)
            data_file = self.processed_folder / self.data_files[partition]
            with open(data_file, "wb") as f:
                th.save((images, labels), f)

    def _labels_as_torch_tensors(self, dataset: pd.DataFrame):
        """Extract labels from pd.Series and convert into torch.Tensor"""
        labels_np = dataset["majority"].values.astype(np.int)
        if self._include_nf:
            # re-map the original index for NF from 9 to 8
            labels_np[labels_np == 9] = 8
        return th.from_numpy(labels_np)
