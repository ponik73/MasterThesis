from typing import Dict, List, Type

from modelEvaluator.accuracy.metrics.metric import Metric
from modelEvaluator.accuracy.metrics import classification, semantic_segmentation, object_detection
from configurationHandler.schemas import SupportedTasks

TASK_METRIC_MAPPING : Dict[SupportedTasks, List[type[Metric]]] = {
    SupportedTasks.IMAGE_CLASSIFICATION : [
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
    SupportedTasks.SEMANTIC_SEGMENTATION : [
        semantic_segmentation.MeanIntersectionOverUnion,
        semantic_segmentation.DiceScoreMicro,
        semantic_segmentation.DiceScoreMacro,
        # semantic_segmentation.HausdorffDistanceEuclidean
    ],
    SupportedTasks.OBJECT_DETECTION : [
        object_detection.IntersectionOverUnion,
        object_detection.MeanAveragePrecisionMacro
    ]
}

def getMetricsForTask(task: SupportedTasks) -> List[Type[Metric]]:
    return TASK_METRIC_MAPPING.get(task)