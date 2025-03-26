from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Any, List, Dict, Type
from pydantic import BaseModel
from torch.types import Tensor
from pathlib import Path

from configurationHandler.schemas import Dataset, Model
from modelEvaluator.api.device.interface import DeviceInterface
from modelEvaluator.accuracy.metrics.metric import Metric, MetricResult
from modelEvaluator.accuracy.metrics.taskMetrics import getMetricsForTask
from user_script.scripts.align_target_predicted import AlignTargetPredictedScript, AlignTargetPredictedFunction
from user_script.exceptions import FunctionLoadingError

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

        self.align_function = self._load_align_function()

    def _load_align_function(self) -> AlignTargetPredictedFunction:
        required_script_key = "align_target_predicted"
        required_function_name = "align_target_predicted"
        
        script_path = Path(self.required_script_paths.get(required_script_key, None))
        script = AlignTargetPredictedScript(script_path)
        script.load()
        
        align_function : AlignTargetPredictedFunction | None = script.get_function_by_name(required_function_name)
        if align_function is None:
            raise FunctionLoadingError(required_function_name)
        return align_function

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
    def evaluate(self) -> List[MetricResult]:
        """Intialize metrics, execute inference and update metrics. Return computed metrics."""
        pass
