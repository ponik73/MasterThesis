from typing import List, Tuple, Any, Dict
from pathlib import Path
# from datasets import Dataset as hfDataset
import numpy as np
from sys import getsizeof
from torch.utils.data import DataLoader

from .accuracy.evaluator.accuracyEvaluatorFactory import accuracyEvaluatorFactory
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
                print(f'[{self.deviceInterface.name}] Finished evaluation of `{model.name}` and `{dataset.name}`.')
            except Exception as e:
                print(f'[{self.deviceInterface.name}] Could not finish evaluation of `{model.name}` and `{dataset.name}`. Reason: {e}.')

    def _executeRun(self, model: Model, dataset: Dataset):
        # Runs are ordered by models. Upload the next model:
        if model.name != self.currentModelName:
            self.currentModelName = model.name
            uploaded = self.deviceInterface.uploadModel(model.localPath, model.name)
            if not uploaded:
                return

        # # Evaluate latency on device:
        # latencyRawOutput = self.deviceInterface.evaluateLatency(model.name, model.framework)
        # # Parse the output:
        # if not latencyRawOutput:
        #     raise Exception(f'[{self.deviceInterface.name}] Error during latency evaluation.')
        
        # parser = latencyParserFactory(model.framework)
        # latencyResults = parser.parse(latencyRawOutput)
    
        
        SEED = 690
        np.random.seed(SEED)
        # Evaluate accuracy on device:
        accuracyEvaluator = accuracyEvaluatorFactory(model.task)(
            model=model,
            deviceInterface=self.deviceInterface,
            dataset=dataset
        )
        accuracyResults = accuracyEvaluator.evaluate()
        
        # TODO: Write results to database: (TODO maybe new fastapi for db)
        # print(latencyResults)
        print(accuracyResults)
        