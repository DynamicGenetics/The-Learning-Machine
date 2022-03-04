from typing import Tuple, Dict
from dataclasses import dataclass
from dataclasses import field
from numpy.typing import ArrayLike


from datasets.fer import FER


@dataclass
class Embedding:

    data: ArrayLike
    labels: ArrayLike
    partition: str
    classes: Tuple[str] = tuple(FER.classes)
    classes_map: Dict[int, str] = field(init=False, default_factory=dict)
    is_ferplus: bool = False

    def __post_init__(self):
        self.classes_map = {i: c for i, c in enumerate(self.classes)}

    @property
    def emotion_labels(self) -> list[str]:
        return [self.classes_map[y] for y in self.labels]

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape
