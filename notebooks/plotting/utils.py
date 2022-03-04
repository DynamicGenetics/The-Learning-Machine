from typing import List, Dict

import numpy as np
import numpy.typing as npt

from sklearn.neighbors import NearestNeighbors
from .model import Embedding
from datasets.fer import FER
from datasets.ferplus import FERPlus


def calculate_centroids(embedding: Embedding) -> dict[str, npt.ArrayLike]:
    """For each class (i.e. emotion), calculate the coordinates of their corresponding centroids."""
    centroids = dict()
    for emo_idx, emotion in embedding.classes_map.items():
        centroids[emotion] = np.mean(
            embedding.data[embedding.labels == emo_idx, :], axis=0
        )
    return centroids


def get_neighbours(embedding: Embedding, n_neighbours: int = 10):
    centroids = calculate_centroids(embedding)

    neighbours_per_emotion = dict()
    neighbours_per_emotion_idx = dict()

    emotion_to_index_map = {v: k for k, v in embedding.classes_map.items()}

    for emotion, centroid in centroids.items():
        emo_idx = emotion_to_index_map[emotion]
        samples = embedding.data[embedding.labels == emo_idx, :]
        nbrs_samples = list()
        samples_selected = list()
        neigh = NearestNeighbors(n_neighbors=15, radius=0.4)
        neigh.fit(samples)
        try:
            nbrs_idx = neigh.kneighbors(
                [
                    centroid,
                ],
                n_neighbors=n_neighbours,
                return_distance=False,
            )
        except ValueError:
            print(f"Neighbours {n_neighbours} not enough for {emotion} - SKIPPING")
        else:
            nbrs_samples = np.vstack([samples[nbidx] for nbidx in nbrs_idx])
            # generate absolute indices
            all_samples_idx = np.where(embedding.labels == emo_idx)[0]
            samples_selected = all_samples_idx[nbrs_idx.ravel()]
        finally:
            neighbours_per_emotion[emotion] = nbrs_samples
            neighbours_per_emotion_idx[emotion] = samples_selected
    return neighbours_per_emotion, neighbours_per_emotion_idx


def get_raw_dataset(embedding: Embedding) -> FER:
    partition = embedding.partition
    if embedding.is_ferplus:
        return FERPlus(root="./", download=True, split=partition)
    return FER(root="./", download=True, split=partition)


def collect_images_per_emotion(dataset: FER, emotion_index: int):
    images = list()
    for img, emotion in iter(dataset):
        if emotion != emotion_index:
            continue
        images.append(img)
    return images


def collect_reference_images(
    embedding: Embedding, n_images: int = 10
) -> Dict[str, List["Image"]]:

    fer_raw = get_raw_dataset(embedding=embedding)
    _, neighbours_per_emotion_idx = get_neighbours(
        embedding=embedding, n_neighbours=n_images
    )
    emotion_to_index_map = {v: k for k, v in embedding.classes_map.items()}

    reference_images_per_emotion = dict()
    for emotion, samples_indices in neighbours_per_emotion_idx.items():
        emo_idx = emotion_to_index_map[emotion]
        selected_samples = list()
        for sample_idx in samples_indices:
            image, label = fer_raw[sample_idx]
            assert label == emo_idx
            selected_samples.append(image)
        reference_images_per_emotion[emotion] = selected_samples
    return reference_images_per_emotion
