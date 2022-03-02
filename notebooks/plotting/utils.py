import numpy as np
import numpy.typing as npt

from sklearn.neighbors import NearestNeighbors
from .model import Embedding
from datasets.fer import FER


def calculate_centroids(embedding: Embedding) -> dict[str, npt.ArrayLike]:
    """For each class (i.e. emotion), calculate the coordinates of their corresponding centroids."""
    centroids = dict()
    for emo_idx, emotion in Embedding.CLASS_MAP.items():
        centroids[emotion] = np.mean(
            embedding.data[embedding.labels == emo_idx, :], axis=0
        )
    return centroids


def get_neighbours(embedding: Embedding, n_neighbours: int = 10):
    centroids = calculate_centroids(embedding)

    neighbours_per_emotion = dict()
    neighbours_per_emotion_idx = dict()

    emotion_to_index_map = {v: k for k, v in Embedding.CLASS_MAP.items()}

    for emotion, centroid in centroids.items():
        emo_idx = emotion_to_index_map[emotion]
        samples = embedding.data[embedding.labels == emo_idx, :]
        neigh = NearestNeighbors(n_neighbors=15, radius=0.4)
        neigh.fit(samples)

        nbrs_idx = neigh.kneighbors(
            [
                centroid,
            ],
            n_neighbors=n_neighbours,
            return_distance=False,
        )
        nbrs_samples = np.vstack([samples[nbidx] for nbidx in nbrs_idx])

        neighbours_per_emotion[emotion] = nbrs_samples
        neighbours_per_emotion_idx[emotion] = nbrs_idx.ravel()
    return neighbours_per_emotion, neighbours_per_emotion_idx


def get_raw_dataset(partition: str) -> FER:
    return FER(root="./", download=True, split=partition)


def collect_images_per_emotion(dataset, emotion_index):
    images = list()
    for img, emotion in iter(dataset):
        if emotion != emotion_index:
            continue
        images.append(img)
    return images


def collect_reference_images(embedding: Embedding) -> dict[str, list["Image"]]:

    fer_raw = get_raw_dataset(partition=embedding.partition)
    _, neighbours_per_emotion_idx = get_neighbours(embedding=embedding)
    emotion_to_index_map = {v: k for k, v in Embedding.CLASS_MAP.items()}

    reference_images_per_emotion = dict()
    for emotion, samples_indices in neighbours_per_emotion_idx.items():
        emo_idx = emotion_to_index_map[emotion]
        samples = collect_images_per_emotion(fer_raw, emo_idx)
        selected_samples = list()
        for sample_idx in samples_indices:
            selected_samples.append(samples[sample_idx])
        reference_images_per_emotion[emotion] = selected_samples
    return reference_images_per_emotion
