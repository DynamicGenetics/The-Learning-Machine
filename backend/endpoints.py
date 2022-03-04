from io import BytesIO
import torch as th
from typing import Sequence, List
from numpy.typing import ArrayLike

from starlette.responses import StreamingResponse

from datasets import Sample, get_dataset
from models import get_model
from models.learning_machine import Prediction
from schemas import Node, EmotionLink, BackendResponse, Annotation
from schemas import ImageBoard, Metric
from settings import LEARNING_MACHINE_MODEL, PRETRAINED
from settings import DATASET_NAME, METRIC_DATASET_NAME
from sklearn.metrics import accuracy_score


def make_nodes(
    samples: Sequence[Sample], predicted_emotions: Prediction, classes: Sequence[str]
) -> List[Node]:
    nodes = list()

    # Check that input classes, and predicted classes are of the same length.
    # If that's not the case (e.g. VGGFERNet trained on 8 out of 10 classes in FER+)
    # retain labels up to a total of `predicted_classes`
    predicted_classes = len(predicted_emotions[0])
    if len(classes) > predicted_classes:
        classes = classes[:predicted_classes]  # retain up to the predicted classes
    for sample, sample_emotions in zip(samples, predicted_emotions):
        emotion_map = {emo: prob for emo, prob in zip(classes, sample_emotions)}
        # Pop Neutral (and Contempt - only for FER+) emotion(s)
        if "neutral" in emotion_map:
            emotion_map.pop("neutral")
        if "contempt" in emotion_map:
            emotion_map.pop("contempt")

        norm = sum(emotion_map.values())

        emotion_map = {emo: prob / norm for emo, prob in emotion_map.items()}
        links = [
            EmotionLink(source=sample.uuid, value=prob, target=emotion)
            for emotion, prob in emotion_map.items()
        ]
        node = Node(
            id=sample.uuid,
            image=f"http://localhost:8000/faces/image/{sample.uuid}",
            links=links,
        )
        nodes.append(node)
    return nodes


def make_metrics(y_true: ArrayLike, emotions_logits: ArrayLike) -> Sequence[Metric]:
    metrics = list()
    _, y_pred = th.max(emotions_logits, 1)
    acc = accuracy_score(y_true, y_pred.detach().cpu().numpy())
    metrics.append(Metric(value=acc))
    return metrics


async def faces(number_of_faces: int = 25, pretrained: bool = PRETRAINED):
    machine = get_model(LEARNING_MACHINE_MODEL, pretrained=pretrained)
    dataset = get_dataset(DATASET_NAME)
    samples = dataset.get_random_samples(k=number_of_faces)
    emotions = machine.predict(samples=samples)
    nodes = make_nodes(samples, emotions, classes=dataset.emotions)
    metric_dataset = get_dataset(METRIC_DATASET_NAME)
    eval_emotions = machine.predict(
        samples=metric_dataset.evaluation_samples, as_proba=False
    )
    metrics = make_metrics(
        y_true=metric_dataset.ground_truth, emotions_logits=eval_emotions
    )
    response = BackendResponse(nodes=nodes, metrics=metrics)
    return response.dict()


async def test_face(image_id: str):
    dataset = get_dataset(DATASET_NAME)
    machine = get_model(LEARNING_MACHINE_MODEL)
    test_sample = dataset[image_id]
    emotions = machine.predict(samples=test_sample)
    response = backend_response([test_sample], emotions, dataset.emotions)
    return response.dict()


async def get_face(image_id: str):
    dataset = get_dataset(DATASET_NAME)
    sample = dataset[image_id]
    image = sample.image
    buffer = BytesIO()
    image.save(buffer, format="png")
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="image/png",
    )


async def annotate(annotation: Annotation):
    dataset = get_dataset(DATASET_NAME)
    machine = get_model(LEARNING_MACHINE_MODEL)
    emotion = annotation.label
    if emotion == "not-human":
        dataset.discard_sample(annotation.image_id)
    else:
        annotated_sample = dataset[annotation.image_id]
        # TODO: this should go in the DB too!!
        annotated_sample.emotion = dataset.emotion_index(emotion)
        machine.fit((annotated_sample,))

    other_samples = [dataset[nid] for nid in annotation.current_nodes]
    other_samples += dataset.get_random_samples(k=annotation.new_nodes)
    updated_emotions = machine.predict(samples=other_samples)
    # Backend Response
    nodes = make_nodes(other_samples, updated_emotions, classes=dataset.emotions)
    metric_dataset = get_dataset(METRIC_DATASET_NAME)
    eval_emotions = machine.predict(
        samples=metric_dataset.evaluation_samples, as_proba=False
    )
    metrics = make_metrics(
        y_true=metric_dataset.ground_truth, emotions_logits=eval_emotions
    )
    response = BackendResponse(nodes=nodes, metrics=metrics)
    return response.dict()


async def forget(current_nodes: ImageBoard, pretrained: bool = False):
    dataset = get_dataset(DATASET_NAME)
    machine = get_model(LEARNING_MACHINE_MODEL, force_init=True, pretrained=pretrained)
    samples = [dataset[nid] for nid in current_nodes.nodes]
    predicted_emotions = machine.predict(samples=samples)
    # Backend Response
    nodes = make_nodes(samples, predicted_emotions, classes=dataset.emotions)
    metric_dataset = get_dataset(METRIC_DATASET_NAME)
    eval_emotions = machine.predict(
        samples=metric_dataset.evaluation_samples, as_proba=False
    )
    metrics = make_metrics(
        y_true=metric_dataset.ground_truth, emotions_logits=eval_emotions
    )
    response = BackendResponse(nodes=nodes, metrics=metrics)
    return response.dict()


async def discard_image(image_id: str):
    dataset = get_dataset(DATASET_NAME)
    machine = get_model(LEARNING_MACHINE_MODEL)
    dataset.discard_sample(image_id)
    new_sample = dataset.get_random_samples(k=1)
    models_preds = machine.predict(samples=new_sample)
    # Backend Response
    nodes = make_nodes(new_sample, models_preds, classes=dataset.emotions)
    metric_dataset = get_dataset(METRIC_DATASET_NAME)
    eval_emotions = machine.predict(
        samples=metric_dataset.evaluation_samples, as_proba=False
    )
    metrics = make_metrics(
        y_true=metric_dataset.ground_truth, emotions_logits=eval_emotions
    )
    response = BackendResponse(nodes=nodes, metrics=metrics)
    return response.dict()


async def serialise_on_shutdown():
    dataset = get_dataset(DATASET_NAME)
    dataset.serialise_session()
