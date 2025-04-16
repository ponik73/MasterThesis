from enum import StrEnum
from pathlib import Path

from datasets import Dataset as HuggingFaceDataset
from pydantic import BaseModel


class SupportedModelHub(StrEnum):
    """Model hubs supported by the framework."""

    KAGGLE = "kaggle"
    HUGGING_FACE = "hf"


class SupportedTasks(StrEnum):
    """Framework supported model tasks."""

    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"


class SupportedModelFrameworks(StrEnum):
    """Machine learning libraries supported by the framework."""

    TFLITE = "tflite"


class _DeviceCredentials(BaseModel):
    """Credential used for ssh connection to a device."""

    user_name: str = ""
    password: str = ""


class Device(BaseModel):
    """Embedded device representation."""

    name: str
    uri: str
    credentials: _DeviceCredentials = _DeviceCredentials()
    sys_name: str | None = None
    kernel_version: str | None = None


class Model(BaseModel):
    """Neural network representation."""

    name: str
    task: SupportedTasks
    platform: SupportedModelHub
    uri: str
    local_path: Path | None = None
    framework: SupportedModelFrameworks | None = None


class Dataset(BaseModel):
    """Dataset representation."""

    name: str
    platform: SupportedModelHub
    uri: str
    dataset_keys: dict[str, str]
    dataset: HuggingFaceDataset | None = None

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class Run(BaseModel):
    """Tuple consisting of an embedded device, a model, a dataset, and a user script."""

    device_name: str
    model_name: str
    dataset_name: str
    required_script_paths: dict[str, str]


class Config(BaseModel):
    """Experiment execution configuration."""

    datasets: list[Dataset]
    devices: list[Device]
    models: list[Model]
    runs: list[Run]
