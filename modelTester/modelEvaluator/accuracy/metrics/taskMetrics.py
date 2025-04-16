from configurationHandler.schemas import SupportedTasks
from modelEvaluator.accuracy.metrics import classification, object_detection, semantic_segmentation
from modelEvaluator.accuracy.metrics.metric import Metric

TASK_METRIC_MAPPING: dict[SupportedTasks, list[type[Metric]]] = {
    SupportedTasks.IMAGE_CLASSIFICATION: [
        classification.AccuracyMacro,
        classification.AccuracyMicro,
        classification.RecallMacro,
        classification.RecallMicro,
        classification.F1Macro,
        classification.F1Micro,
        classification.PrecisionMacro,
        classification.PrecisionMicro,
        classification.SpecificityMacro,
        classification.SpecificityMicro,
    ],
    SupportedTasks.SEMANTIC_SEGMENTATION: [
        semantic_segmentation.MeanIntersectionOverUnion,
        semantic_segmentation.DiceScoreMicro,
        semantic_segmentation.DiceScoreMacro,
    ],
    SupportedTasks.OBJECT_DETECTION: [
        object_detection.IntersectionOverUnion,
        object_detection.MeanAveragePrecisionMacro,
    ],
}


def get_metrics_for_task(task: SupportedTasks) -> list[type[Metric]]:
    """Get list of the metrics evaluated for given task."""
    return TASK_METRIC_MAPPING.get(task)
