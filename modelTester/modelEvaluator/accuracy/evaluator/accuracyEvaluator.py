from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Any, List

from configurationHandler.configurations import Dataset, Model
from modelEvaluator.device.interface import DeviceInterface
from modelEvaluator.accuracy.metrics.metric import Metric

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

        self.predictions : NDArray[Any] | None = None
        self.targets : NDArray[Any] | None = None

        # Get model input size from the API:
        self.modelInputs = deviceInterface.getModelInputInfo(
            modelCustomName=model.name,
            modelFramework=model.framework
        )

    @abstractmethod
    def _metrics(self) -> List[Metric]:
        pass

    @abstractmethod
    def _predict(self):
        """Execute inference and store predictions and ground truths into class variables."""
        pass

    @abstractmethod
    def _calculateMetrics(self):
        pass

    @abstractmethod
    def _task(self) -> Model.SupportedTasks:
        pass

    def evaluate(self):
        self._predict()
        return (self._task().value, self._calculateMetrics())