from __future__ import annotations
from typing import List, Tuple
from pydantic import BaseModel

from modelEvaluator.accuracy.evaluator.accuracyEvaluatorFactory import accuracyEvaluatorFactory
from modelEvaluator.accuracy.metrics.metric import MetricResult
from modelEvaluator.api.device.interface import DeviceInterface
from modelEvaluator.latency.parserFactory import latencyParserFactory
from configurationHandler.schemas import Model, Dataset

class RunPipeline():
    def __init__(
            self,
            deviceInterface: DeviceInterface,
            pairsModelDataset: List[Tuple[Model, Dataset]]
    ):
        self.deviceInterface = deviceInterface
        self.pairsModelDataset = pairsModelDataset

        self.currentModelName : str | None = None

    def empty(self) -> bool:
        return len(self.pairsModelDataset) == 0
    
    def execute_run(self):
        model, dataset = self.pairsModelDataset.pop()
        return (model, dataset, self._executeRun(model, dataset))

    def _executeRun(self, model: Model, dataset: Dataset) -> Tuple[List[MetricResult], List[MetricResult]]:
        # Runs are ordered by models. Upload the next model:
        if model.name != self.currentModelName:
            self.currentModelName = model.name
            uploaded = self.deviceInterface.uploadModel(model.localPath, model.name)
            if not uploaded:
                raise Exception("TODO")

        # # Evaluate latency on device:
        # latencyRawOutput = self.deviceInterface.evaluateLatency(model.name, model.framework)
        # # Parse the output:
        # if not latencyRawOutput:
        #     raise Exception(f'[{self.deviceInterface.name}] Error during latency evaluation.')
        
        # parser = latencyParserFactory(model.framework)
        # latencyResults = parser.parse(latencyRawOutput)
        ########################################################
        #  dummy:
        from modelEvaluator.accuracy.metrics.metric import MetricResult
        latencyResults = [
                MetricResult(name="latency_metric_1", value=1.0),
                MetricResult(name="latency_metric_2", value=2.0)
        ]
        ########################################################
    
        # Evaluate accuracy on device:
        accuracyEvaluator = accuracyEvaluatorFactory(model.task)(
            model=model,
            deviceInterface=self.deviceInterface,
            dataset=dataset
        )
        accuracyResults = accuracyEvaluator.evaluate()
        
        return (latencyResults, accuracyResults)
        