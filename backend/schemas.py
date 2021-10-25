"""Schema definitions for ReSTful APIs based on `pydantic`"""

from pydantic import BaseModel
from typing import List


class EmotionLink(BaseModel):
    source: str
    target: str
    value: float


class Node(BaseModel):
    id: str
    image: str
    links: List[EmotionLink]
    group: str = "data"


class BackendResponse(BaseModel):
    # session_id: str
    nodes: List[Node]


class Annotation(BaseModel):
    image_id: str
    label: str
    current_nodes: List[str]
    new_nodes: int = 1

class ImageBoard(BaseModel):
    nodes: List[str]
