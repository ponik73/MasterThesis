from pydantic import BaseModel
from typing import List
from abc import ABC, abstractmethod

class LatencyMetric(BaseModel):
    """Single latency metric."""
    name: str 
    value: float
    unit: str

class LatencyAssessmentResult(BaseModel):
    """The latency parser output containing latency metrics"""
    metrics: List[LatencyMetric]

    def addMetric(self, metric: LatencyMetric):
        """Method to add a metric to the output."""
        self.metrics.append(metric)

class LatencyParser(ABC):
    @abstractmethod
    def parse(self, data: str) -> LatencyAssessmentResult:
        """Abstract method to parse latency data."""
        pass