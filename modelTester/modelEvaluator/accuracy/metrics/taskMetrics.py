from typing import Dict, List, Type

from modelEvaluator.accuracy.metrics.metric import Metric
from modelEvaluator.accuracy.metrics import classification
from modelEvaluator.accuracy.metrics import semantic_segmentation
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
        semantic_segmentation.HausdorffDistanceEuclidean
    ],
    SupportedTasks.OBJECT_DETECTION : [
        
    ]
}

def getMetricsForTask(task: SupportedTasks) -> List[Type[Metric]]:
    metrics = TASK_METRIC_MAPPING.get(task)
    if not metrics or len(metrics) < 1:
        raise ValueError("TODO")
    return metrics