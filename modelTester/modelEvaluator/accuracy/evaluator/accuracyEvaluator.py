from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Any, List, Dict, Type
from pydantic import BaseModel

from configurationHandler.configurations import Dataset, Model
from modelEvaluator.api.device.interface import DeviceInterface
from modelEvaluator.accuracy.metrics.metric import Metric, MetricResult
from modelEvaluator.accuracy.metrics.taskMetrics import getMetricsForTask

class AccuracyEvaluator(ABC):
    """Abstract base class for accuracy evaluators."""
    BATCH_SIZE = 64
    
    def __init__(
        self,
        model: Model,
        deviceInterface: DeviceInterface,
        dataset: Dataset
    ):
        self.model = model
        self.deviceInterface = deviceInterface
        self.dataset = dataset

        self.predictions : NDArray[Any] | None = None # Class variable for predictions
        self.targets : NDArray[Any] | None = None # Class variable for ground truths

        # Get model input size from the API:
        self.modelInputs = deviceInterface.getModelInputInfo(
            modelCustomName=model.name,
            modelFramework=model.framework
        )

    def _metrics(self) -> List[Type[Metric]]:
        return getMetricsForTask(self.model.task)

    @abstractmethod
    def _predict(self):
        """Execute inference and store predictions and ground truths into class variables."""
        pass

    @abstractmethod
    def _calculateMetrics(self) -> List[MetricResult]:
        pass

    def evaluate(self):
        self._predict()
        return self._calculateMetrics()
    
def computeMetrics(metrics: List[Type[Metric]], predictions: NDArray, targets: NDArray) -> List[MetricResult]:
    """Computes all defined metrics and returns them as a dictionary."""
    
    results = []
    for m in metrics:
        metric = m()
        results.append(
            MetricResult(
                name=metric.name(),
                value=metric.compute(predictions, targets)
            )
        )

    return results