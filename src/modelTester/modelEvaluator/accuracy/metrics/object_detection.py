from typing import Literal

from torchmetrics.detection.iou import IntersectionOverUnion as TorchMetricsIntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .metric import Metric


class IntersectionOverUnion(Metric):
    """Intersection over union object detection metric."""

    def __init__(self, box_format: Literal["xyxy", "xywh", "cxcywh"]) -> None:
        """Metric constructor."""
        self.torch_metric = TorchMetricsIntersectionOverUnion(box_format=box_format)

    def name(self) -> str:
        """Get the name of a metric."""
        return "intersection-over-union"

    def compute(self) -> float:
        """Compute metric value."""
        return self.torch_metric.compute()["iou"]


class MeanAveragePrecisionMacro(Metric):
    """Mean average precision object detection metric."""

    def __init__(self, box_format: Literal["xyxy", "xywh", "cxcywh"]) -> None:
        """Metric constructor."""
        self.torch_metric = MeanAveragePrecision(box_format=box_format, average="macro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "mean-average-precision-macro"

    def compute(self) -> float:
        """Compute metric value."""
        return float(self.torch_metric.compute()["map"])
