from dataclasses import dataclass
from numpy.typing import ArrayLike


from datasets.fer import FER


@dataclass
class Embedding:
    CLASSES = FER.classes
    CLASS_MAP = FER.classes_map()

    data: ArrayLike
    labels: ArrayLike
    partition: str

    @property
    def emotion_labels(self) -> list[str]:
        return [self.CLASS_MAP[y] for y in self.labels]

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape
