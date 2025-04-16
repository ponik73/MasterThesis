from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configurationHandler.schemas import Dataset, Model
    from modelEvaluator.accuracy.metrics.metric import MetricResult
    from modelEvaluator.api.device.interface import DeviceInterface

from modelEvaluator.accuracy.evaluator.accuracyEvaluatorFactory import accuracy_evaluator_factory
from modelEvaluator.exceptions import ModelUploadError
from modelEvaluator.latency.parserFactory import latency_parser_factory


class RunPipeline:
    """Representation of pipeline of evaluation runs for a specific device.

    This class manages execution of model evaluations (both latency and accuracy)
    on a given device using the associated dataset and evaluation scripts.
    """

    def __init__(
        self, device_interface: DeviceInterface, run_settings: list[tuple[Model, Dataset, dict[str, str]]]
    ) -> None:
        """Initialize the run pipeline with a device interface and a list of run configurations.

        :param device_interface: Interface to communicate with the target device.
        :param run_settings: List of tuples containing (Model, Dataset, Required Script Paths).
        """
        self.device_interface = device_interface
        self.run_settings = run_settings

        self.currentModelName: str | None = None

    def empty(self) -> bool:
        """Check if there are no more runs left in the pipeline."""
        return len(self.run_settings) == 0

    def execute_run(self) -> tuple[Model, Dataset, tuple[list[MetricResult], list[MetricResult]]]:
        """Pop the next run from the pipeline and execute it."""
        model, dataset, script_paths = self.run_settings.pop()
        return (model, dataset, self._execute_run(model, dataset, script_paths))

    def _execute_run(
        self, model: Model, dataset: Dataset, script_paths: dict[str, str]
    ) -> tuple[list[MetricResult], list[MetricResult]]:
        # Runs are ordered by models. Upload the next model:
        if model.name != self.currentModelName:
            self.currentModelName = model.name
            uploaded = self.device_interface.upload_model(model.local_path, model.name)
            if not uploaded:
                raise ModelUploadError(model_local_path=model.local_path, api_url=self.device_interface.device.uri)

        # Evaluate latency on device:
        latency_raw_output = self.device_interface.evaluate_latency(model.name, model.framework)
        # Parse the output:
        parser = latency_parser_factory(model.framework)
        latency_results = parser.parse(latency_raw_output)

        # Evaluate accuracy on device:
        accuracy_evaluator = accuracy_evaluator_factory(model.task)(
            model=model, device_interface=self.device_interface, dataset=dataset, required_script_paths=script_paths
        )
        accuracy_results = accuracy_evaluator.evaluate()

        return (latency_results, accuracy_results)
