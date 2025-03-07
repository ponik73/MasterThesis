from abc import ABC, abstractmethod
from typing import Set, Dict, List
from numpy.typing import NDArray

class Metric(ABC):
    """Abstract class for all evaluation metrics."""
    
    @abstractmethod
    def compute(self, predictions: NDArray, targets: NDArray) -> float:
        """Computes the metric value."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the metric."""
        pass

class EvaluationMetrics():
    """Class for metrics computation."""
    
    def compute(self, metrics: List[Metric], predictions: NDArray, targets: NDArray) -> Dict[str, str]:
        """Computes all defined metrics and returns them as a dictionary."""
        results : Dict[str, str] = {}
        for m in metrics:
            results[m.name()] = str(m.compute(predictions, targets))
        return results