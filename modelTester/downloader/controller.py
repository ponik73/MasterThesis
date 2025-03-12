import os
import subprocess
from pathlib import Path
import tempfile
from configurationHandler.configurations import SupportedModelHubs, SupportedModelFrameworks, Model, Dataset
from typing import List, Union, Dict

from huggingface_hub import hf_hub_download
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as hfDataset
from datasets import DatasetDict as hfDatasetDict
import kagglehub

class DownloaderController():
    def __init__(self, models: List[Model], datasets: List[Dataset]):
        try:
            # Sort models and dataset by platform:
            models_by_hub : Dict[SupportedModelHubs, Model] = self.sortByModelHub(models)
            datasets_by_hub : Dict[SupportedModelHubs, Dataset] = self.sortByModelHub(datasets)

            # Create handlers for platforms:
            self.handlerHf = HandlerHF(
                models=models_by_hub[SupportedModelHubs.HUGGING_FACE],
                datasets=datasets_by_hub[SupportedModelHubs.HUGGING_FACE]
            )
            self.handlerKaggle = HandlerKaggle(
                models=models_by_hub[SupportedModelHubs.KAGGLE],
                datasets=datasets_by_hub[SupportedModelHubs.KAGGLE]
            )
        except Exception as ex:
            raise ex
        
    def sortByModelHub(self, items: List[Union[Model, Dataset]]) -> Dict[SupportedModelHubs, Union[Model, Dataset]]:
        group = {}
        for _, modelHub in SupportedModelHubs.__members__.items():
            group[modelHub] = [x for x in items if x.platform == modelHub]
        return group
    
    def download(self):
        modelsHf, datasetsHf = self.handlerHf.download()
        modelsKaggle, datasetsKaggle = self.handlerKaggle.download()
        return [*modelsHf, *modelsKaggle], [*datasetsHf, *datasetsKaggle]
            
class DownloadHandler():
    def __init__(self, models: List[Model], datasets: List[Dataset]):
        self.models = models
        self.datasets = datasets
        self.downloadModelFunc = None
        self.downloadDatasetFunc = None
        
    def download(self) -> tuple[list[Model], list[Dataset]]:
        self.downloadModels()
        self.downloadDatasets()
        return self.models, self.datasets
    
    def downloadModels(self):
        for m in self.models:
            try:
                m.localPath = Path(self.downloadModelFunc(m.uri))
                m.framework = SupportedModelFrameworks(m.localPath.suffix.lstrip("."))
                print(f'{m.name} loaded')
            except Exception as ex:
                m.localPath = None
                print(f'Model "{m.name}" could not be loaded')

    def downloadDatasets(self):
        # If program would consume too much memory, load datasets right before using them
        for d in self.datasets:
            try:
                ds = self.downloadDatasetFunc(d.uri)
                if isinstance(ds, hfDatasetDict):
                    ds = ds[next(iter(ds))]
                if not isinstance(ds, hfDataset):
                    raise Exception
                d.dataset = ds
                print(f'{d.name} loaded')
            except Exception as ex:
                print(f'Dataset "{d.name}" could not be loaded')

class HandlerHF(DownloadHandler):
    hubName = SupportedModelHubs.HUGGING_FACE

    def __init__(self, models, datasets):
        super().__init__(models, datasets)
        self.downloadModelFunc = lambda x: hf_hub_download(**self._parseModelUri(x))
        self.downloadDatasetFunc = self._downloadDataset
    
    def _parseModelUri(self, uri: str) -> tuple[str, str, str]:
        namespace, blob = uri.split("/blob/")
        revision, filename = blob.split("/")
        return {
            "repo_id": namespace,
            "filename": filename,
            "revision": revision
        }
    
    def _downloadDataset(self, uri: str):
        ds = hf_load_dataset(uri)
        return ds
    
class HandlerKaggle(DownloadHandler):
    hubName = SupportedModelHubs.KAGGLE

    def __init__(self, models: list[Model], datasets: list[Dataset]):
        super().__init__(models, datasets)
        self.downloadModelFunc = kagglehub.model_download
        self.downloadDatasetFunc = self._downloadDataset

    def _downloadDataset(self, uri: str) -> hfDataset | hfDatasetDict:
        localPath = kagglehub.dataset_download(uri)
        ds = hf_load_dataset(localPath)
        return ds