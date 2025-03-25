from __future__ import annotations
from typing import List, Tuple, Dict

from modelEvaluator.accuracy.evaluator.accuracyEvaluatorFactory import accuracyEvaluatorFactory
from modelEvaluator.accuracy.metrics.metric import MetricResult
from modelEvaluator.api.device.interface import DeviceInterface
from modelEvaluator.latency.parserFactory import latencyParserFactory
from configurationHandler.schemas import Model, Dataset, Run

class RunPipeline():
    def __init__(
            self,
            deviceInterface: DeviceInterface,
            runSettings: List[Tuple[Model, Dataset, Dict[str, str]]]
            # pairsModelDataset: List[Tuple[Model, Dataset]]
    ):
        self.deviceInterface = deviceInterface
        # self.pairsModelDataset = pairsModelDataset
        self.runSettings = runSettings

        self.currentModelName : str | None = None

    def empty(self) -> bool:
        return len(self.runSettings) == 0
    
    def execute_run(self):
        # model, dataset = self.pairsModelDataset.pop()
        # return (model, dataset, self._executeRun(model, dataset))
        model, dataset, script_paths = self.runSettings.pop()
        return (model, dataset, self._executeRun(model, dataset, script_paths))

    def _executeRun(self, model: Model, dataset: Dataset, script_paths: Dict[str, str]) -> Tuple[List[MetricResult], List[MetricResult]]:
        # Runs are ordered by models. Upload the next model:
        if model.name != self.currentModelName:
            self.currentModelName = model.name
            uploaded = self.deviceInterface.uploadModel(model.localPath, model.name)
            if not uploaded:
                raise Exception("TODO")

        # # Evaluate latency on device:
        latencyRawOutput = self.deviceInterface.evaluateLatency(model.name, model.framework)
        # Parse the output:
        if not latencyRawOutput:
            raise Exception(f'[{self.deviceInterface.name}] Error during latency evaluation.')
        
        parser = latencyParserFactory(model.framework)
        latencyResults = parser.parse(latencyRawOutput)
        ########################################################
        #  dummy:
        # from modelEvaluator.accuracy.metrics.metric import MetricResult
        # latencyResults = [
        #         MetricResult(name="latency_metric_1", value=1.0),
        #         MetricResult(name="latency_metric_2", value=2.0)
        # ]
        ########################################################
    
        # Evaluate accuracy on device:
        accuracyEvaluator = accuracyEvaluatorFactory(model.task)(
            model=model,
            deviceInterface=self.deviceInterface,
            dataset=dataset,
            required_script_paths=script_paths
        )
        accuracyResults = accuracyEvaluator.evaluate()
        
        return (latencyResults, accuracyResults)
        