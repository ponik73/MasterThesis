from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)

from .metric import Metric


class AccuracyMicro(Metric):
    """Micro accuracy classification metric."""

    def __init__(self, num_classes: int) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = MulticlassAccuracy(num_classes=num_classes, average="micro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "accuracy-micro"


class AccuracyMacro(Metric):
    """Macro accuracy classification metric."""

    def __init__(self, num_classes: int) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = MulticlassAccuracy(num_classes=num_classes, average="macro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "accuracy-macro"


class RecallMacro(Metric):
    """Macro recall classification metric."""

    def __init__(self, num_classes: int) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = MulticlassRecall(num_classes=num_classes, average="macro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "recall-macro"


class RecallMicro(Metric):
    """Micro recall classification metric."""

    def __init__(self, num_classes: int) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = MulticlassRecall(num_classes=num_classes, average="micro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "recall-micro"


class F1Macro(Metric):
    """Macro F1 score classification metric."""

    def __init__(self, num_classes: int) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = MulticlassF1Score(num_classes=num_classes, average="macro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "f1-macro"


class F1Micro(Metric):
    """Micro F1 score classification metric."""

    def __init__(self, num_classes: int) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = MulticlassF1Score(num_classes=num_classes, average="micro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "f1-micro"


class PrecisionMacro(Metric):
    """Macro precision classification metric."""

    def __init__(self, num_classes: int) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = MulticlassPrecision(num_classes=num_classes, average="macro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "precision-macro"


class PrecisionMicro(Metric):
    """Micro precision classification metric."""

    def __init__(self, num_classes: int) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = MulticlassPrecision(num_classes=num_classes, average="micro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "precision-micro"


class SpecificityMacro(Metric):
    """Macro specificity classification metric."""

    def __init__(self, num_classes: int) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = MulticlassSpecificity(num_classes=num_classes, average="macro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "specificity-macro"


class SpecificityMicro(Metric):
    """Micro specificity classification metric."""

    def __init__(self, num_classes: int) -> None:
        """Metric constructor."""
        super().__init__()
        self.torch_metric = MulticlassSpecificity(num_classes=num_classes, average="micro")

    def name(self) -> str:
        """Get the name of a metric."""
        return "specificity-micro"
