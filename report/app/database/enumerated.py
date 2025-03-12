from enum import StrEnum
from typing import Union


class SupportedPlatforms(StrEnum):
    KAGGLE = "kaggle"
    HUGGING_FACE = "hf"


class SupportModelFrameworks(StrEnum):
    TFLITE = "tflite"

class SupportedModelTasks(StrEnum):
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"

##################### METRICS

class MetricCategory(StrEnum):
    ACCURACY = "accuracy"
    LATENCY = "latency"

class SupportedMetrics(StrEnum):
    ACCURACY_MICRO = "accuracy-micro"
    ACCURACY_MACRO = "accuracy-macro"
