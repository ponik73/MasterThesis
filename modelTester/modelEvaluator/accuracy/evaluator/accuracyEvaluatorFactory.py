from typing import Dict
from configurationHandler.schemas import SupportedTasks
from .cv.singleInputSingleOutput import SingleInputSingleOutputAccuracyEvaluator
from .cv.object_detection import ObjectDetectionAccuracyEvaluator
from .cv.semantic_segmentation import SemanticSegmentationAccuracyEvaluator

# EVALUATOR_REGISTRY : Dict[Model.SupportedTasks, AccuracyEvaluator] = {
EVALUATOR_REGISTRY = {
    # Model.SupportedTasks.IMAGE_CLASSIFICATION: ImageClassificationAccuracyEvaluator,
    SupportedTasks.IMAGE_CLASSIFICATION: SingleInputSingleOutputAccuracyEvaluator,
    SupportedTasks.OBJECT_DETECTION: ObjectDetectionAccuracyEvaluator, # TODO: implement evaluator
    SupportedTasks.SEMANTIC_SEGMENTATION: SemanticSegmentationAccuracyEvaluator, # TODO: implement evaluator
}

def accuracyEvaluatorFactory(task: SupportedTasks):
    """Returns the corresponding evaluator instance based on model task."""
    evaluatorClass = EVALUATOR_REGISTRY.get(task)
    if not evaluatorClass:
        raise ValueError(f"Model task `{str(task)}` not supported.")
    return evaluatorClass