from typing import List, Tuple
from pathlib import Path
# from datasets import Dataset as hfDataset

from .device.interface import DeviceInterface
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
            model, dataset = self.pairsModelDataset.pop()
            self._executeRun(model, dataset)

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
        # TODO: parser - get by framework + execute
    
        # # Evaluate accuracy on device:
        # TODO: batch?????
        # self.deviceInterface.evaluateAccuracy(self.currentModelName, modelFramework, encodedBatch=None)
        # # TODO: Calculate metrics:

        # TODO: Write results to database: (TODO maybe new fastapi for db)
        
# TODO: batch encoder

# TODO: batch decoder