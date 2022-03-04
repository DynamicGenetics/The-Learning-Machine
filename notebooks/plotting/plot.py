import base64
from io import BytesIO
from typing import List, Tuple

import numpy as np

from bokeh.plotting import figure
from bokeh.palettes import d3
from bokeh.transform import factor_cmap, factor_mark
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool

from matplotlib import pyplot as plt


from datasets.fer import FER

from .utils import calculate_centroids, get_neighbours, get_raw_dataset
from .utils import collect_reference_images
from .model import Embedding

PALETTE = d3["Category10"][10]
MARKERS = [
    "asterisk",
    "circle",
    "diamond",
    "hex",
    "inverted_triangle",
    "plus",
    "square",
    "star",
    "triangle_pin",
    "circle_y",
]

HOVER_TAG = HoverTool(
    tooltips="""
        <div>
            <img src="@image" height="48" alt="@image" width="48" title="@image_label"/>
            <p>@image_label</p>
        </div>
        """
)


def img_to_base64_urls(dataset: FER) -> List[str]:
    """"""
    imgs_base64 = list()
    for img, _ in iter(dataset):
        buffered = BytesIO()
        img.save(buffered, format="png")
        url = "data:image/png;base64,"
        url += base64.b64encode(buffered.getvalue()).decode("utf-8")
        imgs_base64.append(url)
    return imgs_base64


def image_labels(dataset: FER) -> List[str]:
    """"""
    img_labels = list()
    for idx, (_, emotion) in enumerate(iter(dataset)):
        img_labels.append(f"{dataset.classes_map()[emotion]} ({idx})")
    return img_labels


def plot_embedding(
    embedding: Embedding,
    with_hover: bool = True,
    palette: Tuple[str] = PALETTE,
    markers: Tuple[str] = tuple(MARKERS),
    legend_loc: str = "top_right",
):

    fer_raw = get_raw_dataset(embedding=embedding)

    data_source = ColumnDataSource(
        {
            "d1": embedding.data[:, 0],
            "d2": embedding.data[:, 1],
            "emotion": embedding.labels,
            "emotion_label": embedding.emotion_labels,
            "image": img_to_base64_urls(fer_raw),
            "image_label": image_labels(fer_raw),
        }
    )
    p = figure(
        title=f"UMAP projection of Auto-Encoded Faces {embedding.partition.upper()} dataset",
        width=800,
        height=800,
    )
    if with_hover:
        p.add_tools(HOVER_TAG)
    p.scatter(
        x="d1",
        y="d2",
        source=data_source,
        legend_field="emotion_label",
        fill_alpha=0.2,
        size=5,
        color=factor_cmap("emotion_label", palette=palette, factors=embedding.classes),
        marker=factor_mark("emotion_label", markers=markers, factors=embedding.classes),
    )
    p.legend.location = legend_loc
    return p


def plot_centroids(
    embedding: Embedding,
    embedding_fig=None,
):
    centroids = calculate_centroids(embedding)
    centroid_data = ColumnDataSource(
        {
            "d1": [c[0] for c in centroids.values()],
            "d2": [c[1] for c in centroids.values()],
            "emotion_label": list(centroids.keys()),
        }
    )

    if embedding_fig is None:
        embedding_fig = figure(
            title=f"UMAP projection of Centroids of Auto-Encoded Faces in {embedding.partition.upper()} set",
            width=800,
            height=800,
        )

    embedding_fig.scatter(
        x="d1",
        y="d2",
        source=centroid_data,
        fill_alpha=0.8,
        size=8,
        color="black",
        marker=factor_mark("emotion_label", markers=MARKERS, factors=embedding.classes),
    )
    return embedding_fig


def plot_neighbours(embedding: Embedding, n_neighbours: int = 10):
    centroids = calculate_centroids(embedding)
    centroid_data = ColumnDataSource(
        {
            "d1": [c[0] for c in centroids.values()],
            "d2": [c[1] for c in centroids.values()],
            "emotion_label": list(centroids.keys()),
        }
    )
    neighbours_per_emotion, _ = get_neighbours(
        embedding=embedding, n_neighbours=n_neighbours
    )
    neighbours_data = ColumnDataSource(
        {
            "d1": np.hstack([n[:, 0] for n in neighbours_per_emotion.values()]),
            "d2": np.hstack([n[:, 1] for n in neighbours_per_emotion.values()]),
            "emotion_label": [
                emotion
                for emotion, neighbourhood in neighbours_per_emotion.items()
                for _ in range(len(neighbourhood))
            ],
        }
    )
    p = figure(
        title=f"UMAP projection of Auto-Encoded Faces in {embedding.partition.upper()} dataset (with Centroids)",
        width=800,
        height=800,
    )
    p.scatter(
        x="d1",
        y="d2",
        source=neighbours_data,
        fill_alpha=0.8,
        size=8,
        color="magenta",
        legend_field="emotion_label",
        marker=factor_mark("emotion_label", markers=MARKERS, factors=embedding.classes),
    )
    p.scatter(
        x="d1",
        y="d2",
        source=centroid_data,
        fill_alpha=0.8,
        size=8,
        color="black",
        marker=factor_mark("emotion_label", markers=MARKERS, factors=embedding.classes),
    )
    p.legend.location = "bottom_right"
    return p


def plot_reference_images(embedding: Embedding, ncols: int = 10):
    print(f"REFERENCE IMAGES FOR {embedding.partition.upper()} DATASET")
    reference_images_per_emotion = collect_reference_images(
        embedding=embedding, n_images=ncols
    )
    for emotion, selected_samples in reference_images_per_emotion.items():
        print(f"Selected Representatives for Emotion {emotion}")
        _, axs = plt.subplots(
            nrows=1, ncols=ncols, sharex=True, sharey=True, figsize=(25, 3)
        )
        for idx, sample in enumerate(selected_samples):
            axs[idx].imshow(
                sample, interpolation="bilinear", cmap="gist_gray", aspect="auto"
            )
        plt.show()
