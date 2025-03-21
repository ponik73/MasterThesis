from pathlib import Path
from datasets import Dataset as HuggingFaceDataset
from enum import StrEnum
from pydantic import BaseModel
from typing import List, Dict

class SupportedModelHubs(StrEnum):
    KAGGLE = "kaggle"
    HUGGING_FACE = "hf"

class SupportedTasks(StrEnum):
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"

class SupportedModelFrameworks(StrEnum):
    TFLITE = "tflite"

class Device(BaseModel):
    name: str
    uri: str
    fingerprint : str | None = None
    distinctAttributes : List[str] = ["name"]


class Model(BaseModel):
    name: str
    task: SupportedTasks
    platform: SupportedModelHubs
    uri: str
    localPath: Path | None = None
    framework : SupportedModelFrameworks | None = None
    distinctAttributes : List[str] = ["name"]


class Dataset(BaseModel):
    name: str
    platform: SupportedModelHubs
    uri: str
    dataset_keys: Dict[str, str]
    dataset : HuggingFaceDataset | None = None
    distinctAttributes : List[str] = ["name"]
    class Config:
        arbitrary_types_allowed = True

class Run(BaseModel):
    deviceName: str
    modelName: str
    datasetName: str


class Config(BaseModel):
    datasets: List[Dataset]
    devices: List[Device]
    models: List[Model]
    runs: List[Run]
