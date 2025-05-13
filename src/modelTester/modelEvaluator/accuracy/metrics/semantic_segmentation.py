from typing import Literal

from torchmetrics.segmentation import DiceScore, MeanIoU

from .metric import Metric


class MeanIntersectionOverUnion(Metric):
    """Mean intersection over union segmentation metric."""

    def __init__(self, num_classes: int, input_format: Literal["one-hot", "index"]) -> None:
        """Metric constructor."""
        self.torch_metric = MeanIoU(num_classes=num_classes, input_format=input_format)

    def name(self) -> str:
        """Get the name of a metric."""
        return "mean-intersection-over-union"


class DiceScoreMicro(Metric):
    """Micro dice score segmentation metric."""

    def __init__(self, num_classes: int, input_format: Literal["one-hot", "index"]) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = DiceScore(num_classes=num_classes, input_format=input_format, average="micro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "dice-score-micro"


class DiceScoreMacro(Metric):
    """Macro dice score segmentation metric."""

    def __init__(self, num_classes: int, input_format: Literal["one-hot", "index"]) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = DiceScore(num_classes=num_classes, input_format=input_format, average="micro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "dice-score-macro"
