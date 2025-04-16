from abc import ABC, abstractmethod
from pathlib import Path

from configurationHandler.schemas import Dataset, Model
from modelEvaluator.accuracy.metrics.metric import Metric, MetricResult
from modelEvaluator.accuracy.metrics.taskMetrics import get_metrics_for_task
from modelEvaluator.api.device.interface import DeviceInterface
from torch.types import Tensor
from user_script.exceptions import FunctionLoadingError
from user_script.scripts.align_target_predicted import AlignTargetPredictedFunction, AlignTargetPredictedScript


class AccuracyEvaluator(ABC):
    """Abstract base class for accuracy evaluators."""

    BATCH_SIZE = 64

    def __init__(
        self, model: Model, device_interface: DeviceInterface, dataset: Dataset, required_script_paths: dict[str, str]
    ) -> None:
        """Initialize the evaluator with model, device interface, dataset and script paths.

        :param model: The model to be evaluated.
        :param device_interface: Interface to interact with the embedded device.
        :param dataset: The dataset to evaluate the model on.
        :param required_script_paths: Dictionary mapping script names to their file paths.
        """
        self.model = model
        self.device_interface = device_interface
        self.dataset = dataset
        self.required_script_paths = required_script_paths

        # Get model input size from the API:
        self.modelInputs = device_interface.get_model_input_info(
            model_custom_name=model.name, model_framework=model.framework
        )

        self.metrics: list[Metric] = []

        self.align_function = self._load_align_function()

    def _load_align_function(self) -> AlignTargetPredictedFunction:
        required_script_key = "align_target_predicted"
        required_function_name = "align_target_predicted"

        script_path = Path(self.required_script_paths.get(required_script_key, None))
        script = AlignTargetPredictedScript(script_path)
        script.load()

        align_function: AlignTargetPredictedFunction | None = script.get_function_by_name(required_function_name)
        if align_function is None:
            raise FunctionLoadingError(required_function_name)
        return align_function

    def _metrics(self) -> list[type[Metric]]:
        return get_metrics_for_task(self.model.task)

    def init_metrics(self, **kwargs) -> None:
        """Instantiate metric objects with given parameters."""
        for m in self._metrics():
            self.metrics.append(m(**kwargs))

    def update_metrics(self, predictions: Tensor, targets: Tensor) -> None:
        """Update all metrics with a new batch of predictions and targets."""
        for m in self.metrics:
            m.update(predictions, targets)

    def compute_metrics(self) -> list[MetricResult]:
        """Compute all defined metrics and returns them as a dictionary."""
        return [MetricResult(name=m.name(), value=m.compute()) for m in self.metrics]

    @abstractmethod
    def evaluate(self) -> list[MetricResult]:
        """Intialize metrics, execute inference and update metrics. Return computed metrics."""
