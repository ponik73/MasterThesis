from configurationHandler.schemas import SupportedTasks

from .cv.classificationAccuracyEvaluator import ClassificationAccuracyEvaluator
from .cv.object_detection import ObjectDetectionAccuracyEvaluator
from .cv.semantic_segmentation import SemanticSegmentationAccuracyEvaluator

EVALUATOR_REGISTRY = {
    SupportedTasks.IMAGE_CLASSIFICATION: ClassificationAccuracyEvaluator,
    SupportedTasks.OBJECT_DETECTION: ObjectDetectionAccuracyEvaluator,
    SupportedTasks.SEMANTIC_SEGMENTATION: SemanticSegmentationAccuracyEvaluator,
}


def accuracy_evaluator_factory(
    task: SupportedTasks,
) -> ClassificationAccuracyEvaluator | ObjectDetectionAccuracyEvaluator | SemanticSegmentationAccuracyEvaluator:
    """Return the corresponding evaluator instance based on model task."""
    evaluator_class = EVALUATOR_REGISTRY.get(task)
    if not evaluator_class:
        error_message = f"Model task `{task}` not supported."
        raise ValueError(error_message)
    return evaluator_class
