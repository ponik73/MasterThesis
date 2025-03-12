from pathlib import Path
from datasets import Dataset as hfDataset
from enum import StrEnum
from pydantic import BaseModel
from typing import List

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
    dataset : hfDataset | None = None
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

# supportedModelHubs = [
#         "hf",
#         "kaggle"
#     ]

# # Classes representing configuration file objects:
# class Device():
#     attributes = ["name", "uri"]
#     distinctAttributes = ["name", "uri"]

#     def __init__(self, name: str, uri: str):
#         self.name = name
#         self.uri = uri
#         self.fingerprint : str | None = None

# class Model():
#     attributes = ["name", "task", "platform", "uri"]
#     distinctAttributes = ["name", "uri"]

#     class SupportedTasks(StrEnum):
#         # IMAGE_CLASSIFICATION_BINARY = "image-classification-binary"
#         # IMAGE_CLASSIFICATION_MULTI_CLASS = "image-classification-multi-class"
#         # IMAGE_CLASSIFICATION_MULTI_LABEL = "image-classification-multi-label"
#         IMAGE_CLASSIFICATION = "image-classification"
#         OBJECT_DETECTION = "object-detection"
#         SEMANTIC_SEGMENTATION = "semantic-segmentation"
    
#     class SupportedFrameworks(StrEnum):
#         TFLITE = "tflite"

#     def __init__(self, name: str, task: str, platform: str, uri: str):
#         self.name = name
#         self.uri = uri

#         # Check if given task is supported:
#         if task not in self.SupportedTasks.__members__.values():
#             raise ValueError(f'Task "{task}" is not supported. Supported tasks are: {self.SupportedTasks.__members__.values()}')
#         self.task = self.SupportedTasks(task)

#         # Check if given model hub is supported:
#         if platform not in supportedModelHubs:
#             raise ValueError(f'Model hub "{platform}" is not supported. Supported platforms are: {supportedModelHubs}')
#         self.platform = platform

#         self.localPath: Path|None = None
#         self.framework : Model.SupportedFrameworks | None = None
        
# class Dataset():
#     attributes = ["name", "platform", "uri"]
#     distinctAttributes = ["name"]

#     def __init__(self, name: str, platform: str, uri: str):
#         self.name = name
#         self.uri = uri

#         # Check if given model hub is supported:
#         if platform not in supportedModelHubs:
#             raise ValueError(f'Model hub "{platform}" is not supported. Supported platforms are: {supportedModelHubs}')
#         self.platform = platform

#         self.dataset : hfDataset | None = None
        
# class Run():
#     attributes = ["device", "model", "dataset"]
#     distinctAttributes = []
#     def __init__(self, device: str, model: str, dataset: str):
#         self.deviceName = device
#         self.modelName = model
#         self.datasetName = dataset