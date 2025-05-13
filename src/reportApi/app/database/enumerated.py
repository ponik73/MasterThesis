from enum import StrEnum


class SupportedPlatforms(StrEnum):
    """Enumerated supported model platforms (public storages)."""

    KAGGLE = "kaggle"
    HUGGING_FACE = "hf"


platform_url_mapping = {
    SupportedPlatforms.HUGGING_FACE: "https://huggingface.co",
    SupportedPlatforms.KAGGLE: "https://www.kaggle.com",
}


class SupportModelFrameworks(StrEnum):
    """Enumerated supported neural network frameworks."""

    TFLITE = "tflite"


class SupportedModelTasks(StrEnum):
    """Enumerated supported neural network tasks."""

    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"
    SEMANTIC_SEGMENTATION = "semantic-segmentation"


##################### METRICS


class MetricCategory(StrEnum):
    """Enumerated metric categories."""

    ACCURACY = "accuracy"
    LATENCY = "latency"
