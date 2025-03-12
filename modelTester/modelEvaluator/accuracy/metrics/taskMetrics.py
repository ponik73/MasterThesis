from typing import Dict, List, Type

from modelEvaluator.accuracy.metrics.metric import Metric
from modelEvaluator.accuracy.metrics import classification
from configurationHandler.configurations import Model

TASK_METRIC_MAPPING : Dict[Model.SupportedTasks, List[type[Metric]]] = {
    Model.SupportedTasks.IMAGE_CLASSIFICATION : [
        classification.AccuracyMacro,
        classification.AccuracyMicro,
        classification.RecallMacro,
        classification.RecallMicro
    ]
}

def getMetricsForTask(task: Model.SupportedTasks) -> List[Type[Metric]]:
    metrics = TASK_METRIC_MAPPING.get(task)
    if not metrics or len(metrics) < 1:
        raise ValueError("TODO")
    return metrics