import os
import subprocess
from pathlib import Path
import tempfile
from configurationHandler.configurations import supportedModelHubs, Model, Dataset

from huggingface_hub import hf_hub_download

class DownloaderController():
    def __init__(self, cfg: dict):
        try:
            self.cfg = cfg

            # Sort models and dataset by platform:
            models = self.sortByModelHub("models")
            datasets = self.sortByModelHub("datasets")

            # Create handlers for platforms:
            self.handlerHf = HandlerHF(models["hf"], datasets["hf"])
            self.handlerKaggle = HandlerKaggle(models["kaggle"], datasets["kaggle"])
        except Exception as ex:
            raise ex
        
    def sortByModelHub(self, groupName: str):
        group = {}
        for modelHub in supportedModelHubs:
            group[modelHub] = [x for x in self.cfg[groupName] if x.platform == modelHub]
        return group
    
    def download(self):
        self.handlerHf.download()
        # self.handlerKaggle.download()
        return "downloaded Models info", "downloaded Datasets info"
            
class DownloadHandler():
    def __init__(self, models: list[Model], datasets: list[Dataset]):
        self.models = models
        self.datasets = datasets
        
    def download(self):
        self.downloadModels()
        # self.downloadDatasets()

class HandlerHF(DownloadHandler):
    hubName = supportedModelHubs[0]

    def __init__(self, models: list[Model], datasets: list[Dataset]):
        super().__init__(models, datasets)
    
    def _parseModelUri(self, uri: str) -> tuple[str, str, str]:
        namespace, blob = uri.split("/blob/")
        revision, filename = blob.split("/")
        return namespace, filename, revision
        
    def downloadModels(self):
        for m in self.models:
            namespace, filename, revision = self._parseModelUri(m.uri)
            try:
                m.localPath = Path(hf_hub_download(repo_id=namespace, filename=filename, revision=revision))
            except Exception as ex:
                m.localPath = None
                print(f'Model "{m.name}" has invalid uri "{m.uri}"')

    def downloadDataset():
        #TODO: implement using Datasets.load_dataset(..., streaming=True)
        pass
    
class HandlerKaggle(DownloadHandler):
    hubName = supportedModelHubs[1]

    def __init__(self, models: list[Model], datasets: list[Dataset]):
        super().__init__(models, datasets)
    def downloadModels(self):
        print("Kaggle model download not implemented")
