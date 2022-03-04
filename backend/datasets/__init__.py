"""
This package provides access to all the datasets available to the learning machine.
"""

from .fer import FER
from .sources import load_fer_dataset_lazy, load_fer_training_lazy
from .sources import load_fer_validation_lazy, load_fer_evaluation_ds_lazy
from .sources import load_ferplus_dataset_lazy, load_ferplus_evaluation_ds_lazy
from .sources import DataSource, Sample


# Available Dataset Keys
FER_DATASET = "FER"
FERPLUS_DATASET = "FER+"
FERPLUS_METRICS = "FER+_METRICS"
FER_TRAINING = "FER_TRAIN"
FER_VALIDATION = "FER_VALIDATION"
FER_METRICS = "FER_METRICS"

DATASETS_PROXY = {
    FER_DATASET: load_fer_dataset_lazy(),
    FER_TRAINING: load_fer_training_lazy(),
    FER_VALIDATION: load_fer_validation_lazy(),
    FER_METRICS: load_fer_evaluation_ds_lazy(),
    FERPLUS_DATASET: load_ferplus_dataset_lazy(),
    FERPLUS_METRICS: load_ferplus_evaluation_ds_lazy(),
}


def get_dataset(key: str) -> DataSource:
    """
    Instantiate a DataSource object depending on the selected
    torch Dataset

    Parameters
    ----------
    key : str
        A key for the DataSource Proxy map

    Returns
    -------
    DataSource
        DataSource instance associated to the selected key.

    Raises
    ------
    ValueError
        Raised if the input key is not found in the DataSets proxy map.
    """
    try:
        data_source = DATASETS_PROXY[key]
    except KeyError:
        raise ValueError(f"Invalid Dataset Key: {key}")
    else:
        return data_source


__all__ = [
    "FER",
    "DataSource",
    "FER_DATASET",
    "FER_METRICS",
    "FER_VALIDATION",
    "FER_TRAINING",
    "FERPLUS_METRICS",
    "FERPLUS_DATASET",
    "DATASETS_PROXY",
    "get_dataset",
    "Sample",
]
