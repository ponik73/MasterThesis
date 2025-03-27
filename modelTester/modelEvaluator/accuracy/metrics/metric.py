from abc import ABC, abstractmethod

from pydantic import BaseModel
from torch.types import Tensor
from torchmetrics import Metric as TorchMetric


class MetricResult(BaseModel):
    # """Base model"""
    name: str
    value: float
    unit: str | None = None


class Metric(ABC):
    """Abstract class for all evaluation metrics."""

    def __init__(self) -> None:
        self.torch_metric: TorchMetric | None = None

    def update(self, predictions: Tensor, targets: Tensor) -> float:
        self.torch_metric.update(predictions, targets)

    def compute(self) -> float:
        """Compute the value of a metric."""
        metric_value: Tensor = self.torch_metric.compute()
        return float(metric_value.item())

    @abstractmethod
    def name(self) -> str:
        """Get the name of a metric."""
