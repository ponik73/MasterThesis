from enum import StrEnum
from pathlib import Path

from datasets import Dataset as HuggingFaceDataset
from pydantic import BaseModel


class SupportedModelHubs(StrEnum):
    KAGGLE = "kaggle"
    HUGGING_FACE = "hf"


class SupportedTasks(StrEnum):
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"


class SupportedModelFrameworks(StrEnum):
    TFLITE = "tflite"


class _DeviceCredentials(BaseModel):
    user_name: str = ""
    password: str = ""


class Device(BaseModel):
    name: str
    uri: str
    credentials: _DeviceCredentials
    sys_name: str | None = None
    kernel_version: str | None = None
    distinctAttributes: list[str] = ["name"]


class Model(BaseModel):
    name: str
    task: SupportedTasks
    platform: SupportedModelHubs
    uri: str
    localPath: Path | None = None
    framework: SupportedModelFrameworks | None = None
    distinctAttributes: list[str] = ["name"]


class Dataset(BaseModel):
    name: str
    platform: SupportedModelHubs
    uri: str
    dataset_keys: dict[str, str]
    dataset: HuggingFaceDataset | None = None
    distinctAttributes: list[str] = ["name"]

    class Config:
        arbitrary_types_allowed = True


class Run(BaseModel):
    deviceName: str
    modelName: str
    datasetName: str
    required_script_paths: dict[str, str]


class Config(BaseModel):
    datasets: list[Dataset]
    devices: list[Device]
    models: list[Model]
    runs: list[Run]
