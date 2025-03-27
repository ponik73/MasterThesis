from typing import Literal

from torchmetrics.segmentation import DiceScore, HausdorffDistance, MeanIoU

from .metric import Metric


class MeanIntersectionOverUnion(Metric):
    def __init__(self, num_classes: int, input_format: Literal["one-hot", "index"]) -> None:
        """Metric constructor."""
        self.torch_metric = MeanIoU(num_classes=num_classes, input_format=input_format)

    def name(self) -> str:
        """Get the name of a metric."""
        return "mean-intersection-over-union"


class DiceScoreMicro(Metric):
    def __init__(self, num_classes: int, input_format: Literal["one-hot", "index"]) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = DiceScore(num_classes=num_classes, input_format=input_format, average="micro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "dice-score-micro"


class DiceScoreMacro(Metric):
    def __init__(self, num_classes: int, input_format: Literal["one-hot", "index"]) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = DiceScore(num_classes=num_classes, input_format=input_format, average="micro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "dice-score-macro"


class HausdorffDistanceEuclidean(Metric):
    def __init__(self, num_classes: int, input_format: Literal["one-hot", "index"]) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = HausdorffDistance(
            num_classes=num_classes, input_format=input_format, distance_metric="euclidean"
        )

    def name(self) -> str:
        """Get the name of a metric."""
        return "hausdorff-distance-euclidean"
