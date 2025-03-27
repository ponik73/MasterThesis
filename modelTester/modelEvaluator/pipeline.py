from __future__ import annotations

from configurationHandler.schemas import Dataset, Model

from modelEvaluator.accuracy.evaluator.accuracyEvaluatorFactory import accuracyEvaluatorFactory
from modelEvaluator.accuracy.metrics.metric import MetricResult
from modelEvaluator.api.device.interface import DeviceInterface
from modelEvaluator.exceptions import ModelUploadError
from modelEvaluator.latency.parserFactory import latencyParserFactory


class RunPipeline:
    def __init__(self, deviceInterface: DeviceInterface, runSettings: list[tuple[Model, Dataset, dict[str, str]]]):
        self.deviceInterface = deviceInterface
        self.runSettings = runSettings

        self.currentModelName: str | None = None

    def empty(self) -> bool:
        return len(self.runSettings) == 0

    def execute_run(self):
        model, dataset, script_paths = self.runSettings.pop()
        return (model, dataset, self._executeRun(model, dataset, script_paths))

    def _executeRun(
        self, model: Model, dataset: Dataset, script_paths: dict[str, str]
    ) -> tuple[list[MetricResult], list[MetricResult]]:
        # Runs are ordered by models. Upload the next model:
        if model.name != self.currentModelName:
            self.currentModelName = model.name
            uploaded = self.deviceInterface.upload_model(model.localPath, model.name)
            if not uploaded:
                raise ModelUploadError(model_local_path=model.localPath, api_url=self.deviceInterface.device.uri)

        # Evaluate latency on device:
        latencyRawOutput = self.deviceInterface.evaluateLatency(model.name, model.framework)
        # Parse the output:
        if not latencyRawOutput:
            raise Exception(f"[{self.deviceInterface.name}] Error during latency evaluation.")
        parser = latencyParserFactory(model.framework)
        latencyResults = parser.parse(latencyRawOutput)

        # Evaluate accuracy on device:
        accuracyEvaluator = accuracyEvaluatorFactory(model.task)(
            model=model, deviceInterface=self.deviceInterface, dataset=dataset, required_script_paths=script_paths
        )
        accuracyResults = accuracyEvaluator.evaluate()

        return (latencyResults, accuracyResults)
