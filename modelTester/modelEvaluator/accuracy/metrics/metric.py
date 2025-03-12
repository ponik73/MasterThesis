from abc import ABC, abstractmethod
from typing import Type, Dict, List
from numpy.typing import NDArray
from pydantic import BaseModel

class MetricResult(BaseModel):
    # """Base model"""
    name: str
    value: float
    # TODO: unit

class Metric(ABC):
    """Abstract class for all evaluation metrics."""
    
    @abstractmethod
    def compute(self, predictions: NDArray, targets: NDArray) -> MetricResult:
        """Computes the metric value."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the metric."""
        pass