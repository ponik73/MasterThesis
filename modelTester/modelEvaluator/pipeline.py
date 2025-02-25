from typing import List, Tuple
from pathlib import Path
# from datasets import Dataset as hfDataset

from .device.interface import DeviceInterface
from .latency.parserFactory import latencyParserFactory
from configurationHandler.configurations import Model, Dataset

class RunPipeline():
    def __init__(
            self,
            deviceInterface: DeviceInterface,
            pairsModelDataset: List[Tuple[Model, Dataset]]
    ):
        self.deviceInterface = deviceInterface
        self.pairsModelDataset = pairsModelDataset

        self.currentModelName : str | None = None

    def execute(self):
        while len(self.pairsModelDataset) != 0:
            try:
                model, dataset = self.pairsModelDataset.pop()
                self._executeRun(model, dataset)
                print(f'[{self.deviceInterface.name}] Finished evaluation of `{model.name}` and `{dataset.name}`. Reason: {e}.')
            except Exception as e:
                print(f'[{self.deviceInterface.name}] Could not finish evaluation of `{model.name}` and `{dataset.name}`. Reason: {e}.')

    def _executeRun(self, model: Model, dataset: Dataset):
        # Runs are ordered by models. Upload the next model:
        if model.name != self.currentModelName:
            self.currentModelName = model.name
            uploaded = self.deviceInterface.uploadModel(model.localPath, model.name)
            if not uploaded:
                return
        
        # Framework of the model derived from file extension:
        modelFramework = model.localPath.suffix.lstrip(".")

        # Evaluate latency on device:
        latencyRawOutput = self.deviceInterface.evaluateLatency(model.name, modelFramework)
        # Parse the output:
        if not latencyRawOutput:
            return
        
        parser = latencyParserFactory(modelFramework)
        latencyResults = parser.parse(latencyRawOutput)
    
        # # Evaluate accuracy on device:
        # TODO: batch?????
        # self.deviceInterface.evaluateAccuracy(self.currentModelName, modelFramework, encodedBatch=None)
        # # TODO: Calculate metrics:

        # TODO: Write results to database: (TODO maybe new fastapi for db)
        
# TODO: batch encoder

# TODO: batch decoder