from typing import Dict
from configurationHandler.configurations import Model
from .cv.imageClassification import ImageClassificationAccuracyEvaluator

# EVALUATOR_REGISTRY : Dict[Model.SupportedTasks, AccuracyEvaluator] = {
EVALUATOR_REGISTRY = {
    # Model.SupportedTasks.IMAGE_CLASSIFICATION_BINARY: ImageClassificationAccuracyEvaluator,
    # Model.SupportedTasks.IMAGE_CLASSIFICATION_MULTI_CLASS: ImageClassificationAccuracyEvaluator, 
    # Model.SupportedTasks.IMAGE_CLASSIFICATION_MULTI_LABEL: ImageClassificationAccuracyEvaluator,
    Model.SupportedTasks.IMAGE_CLASSIFICATION: ImageClassificationAccuracyEvaluator,
    # Model.SupportedTasks.OBJECT_DETECTION: AccuracyEvaluator, # TODO: implement evaluator
    # Model.SupportedTasks.SEMANTIC_SEGMENTATION: AccuracyEvaluator, # TODO: implement evaluator
}

def accuracyEvaluatorFactory(task: Model.SupportedTasks):
    """Returns the corresponding evaluator instance based on model task."""
    evaluatorClass = EVALUATOR_REGISTRY.get(task)
    if not evaluatorClass:
        raise ValueError(f"Model task `{str(task)}` not supported.")
    return evaluatorClass