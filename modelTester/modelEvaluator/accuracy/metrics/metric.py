from torch.types import Tensor
from torchmetrics import Metric as TorchMetric
from pydantic import BaseModel
from abc import ABC, abstractmethod

class MetricResult(BaseModel):
    # """Base model"""
    name: str
    value: float
    # TODO: unit?

class Metric(ABC):
    """Abstract class for all evaluation metrics."""
    def __init__(self):
        self.torch_metric : TorchMetric | None = None

    def update(self, predictions: Tensor, targets: Tensor) -> float:
        self.torch_metric.update(predictions, targets)

    def compute(self) -> float:
        """Computes the metric value."""
        metric_value : Tensor = self.torch_metric.compute()
        return float(metric_value.item())

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the metric."""
        pass