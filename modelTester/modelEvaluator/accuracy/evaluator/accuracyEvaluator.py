from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Any, List, Dict, Type
from pydantic import BaseModel
from torch.types import Tensor

from configurationHandler.schemas import Dataset, Model
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
        dataset: Dataset,
        required_script_paths: Dict[str, str]
    ):
        self.model = model
        self.deviceInterface = deviceInterface
        self.dataset = dataset
        self.required_script_paths = required_script_paths

        # Get model input size from the API:
        self.modelInputs = deviceInterface.getModelInputInfo(
            modelCustomName=model.name,
            modelFramework=model.framework
        )

        self.metrics : List[Metric] = []

    def _metrics(self) -> List[Type[Metric]]:
        return getMetricsForTask(self.model.task)
    
    def initMetrics(self, **kwargs):
        for m in self._metrics():
            self.metrics.append(m(**kwargs))

    def updateMetrics(self, predictions: Tensor, targets: Tensor):
        for m in self.metrics:
            m.update(predictions, targets)

    def computeMetrics(self) -> List[MetricResult]:
        """Computes all defined metrics and returns them as a dictionary."""
        
        results = []
        for m in self.metrics:
            results.append(
                MetricResult(
                    name=m.name(),
                    value=m.compute()
                )
            )

        return results
    
    @abstractmethod
    def evaluate(self):
        """Intialize metrics, execute inference and update metrics. Return computed metrics."""
        pass
