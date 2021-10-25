from io import BytesIO
from typing import Sequence, List

from starlette.responses import StreamingResponse

from datasets import Sample, get_dataset
from models import get_model
from models.learning_machine import Prediction
from schemas import Node, EmotionLink, BackendResponse, Annotation
from schemas import ImageBoard
from settings import LEARNING_MACHINE_MODEL, DATASET_NAME


def make_nodes(
    samples: Sequence[Sample], emotions: Prediction, classes: Sequence[str]
) -> List[Node]:
    nodes = list()
    for sample, emotion in zip(samples, emotions):
        emotion_map = {c: p for c, p in zip(classes, emotion)}
        norm = sum(emotion_map.values())
        # Pop Neutral emotion after having calculated weights
        emotion_map.pop("neutral")

        emotion_map = {c: v / norm for c, v in emotion_map.items()}
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


async def faces(number_of_faces: int = 25, pretrained: bool = False):
    machine = get_model(LEARNING_MACHINE_MODEL, pretrained=pretrained)
    dataset = get_dataset(DATASET_NAME)
    samples = dataset.get_random_samples(k=number_of_faces)
    emotions = machine.predict(samples=samples)
    nodes = make_nodes(samples, emotions, dataset.emotions)
    response = BackendResponse(nodes=nodes)
    return response.dict()


async def test_face(image_id: str):
    dataset = get_dataset(DATASET_NAME)
    machine = get_model(LEARNING_MACHINE_MODEL)
    test_sample = dataset[image_id]
    emotions = machine.predict(samples=test_sample)
    nodes = make_nodes([test_sample], emotions, dataset.emotions)
    response = BackendResponse(nodes=nodes)
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
    nodes = make_nodes(other_samples, updated_emotions, dataset.emotions)
    response = BackendResponse(nodes=nodes)
    return response.dict()


async def forget(current_nodes: ImageBoard, pretrained: bool = False):
    dataset = get_dataset(DATASET_NAME)
    machine = get_model(LEARNING_MACHINE_MODEL, force_init=True, pretrained=pretrained)
    samples = [dataset[nid] for nid in current_nodes.nodes]
    predicted_emotions = machine.predict(samples=samples)
    nodes = make_nodes(samples, predicted_emotions, dataset.emotions)
    response = BackendResponse(nodes=nodes)
    return response.dict()

async def discard_image(image_id: str):
    dataset = get_dataset(DATASET_NAME)
    machine = get_model(LEARNING_MACHINE_MODEL)
    dataset.discard_sample(image_id)
    new_sample = dataset.get_random_samples(k=1)
    models_preds = machine.predict(samples=new_sample)
    nodes = make_nodes(new_sample, models_preds, dataset.emotions)
    response = BackendResponse(nodes=nodes)
    return response.dict()


async def serialise_on_shutdown():
    dataset = get_dataset(DATASET_NAME)
    dataset.serialise_session()
