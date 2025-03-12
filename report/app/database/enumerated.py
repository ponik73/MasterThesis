from enum import StrEnum

class SupportedPlatforms(StrEnum):
    KAGGLE = "kaggle"
    HUGGING_FACE = "hf"

platform_url_mapping = {
    SupportedPlatforms.HUGGING_FACE: "https://huggingface.co",
    SupportedPlatforms.KAGGLE: "https://www.kaggle.com"
}

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
