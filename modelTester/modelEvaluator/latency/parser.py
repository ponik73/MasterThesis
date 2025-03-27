from abc import ABC, abstractmethod

from modelEvaluator.accuracy.metrics.metric import MetricResult


class LatencyParser(ABC):
    @abstractmethod
    def parse(self, data: str) -> list[MetricResult]:
        """Abstract method to parse latency data."""
