import os
import subprocess
from pathlib import Path
import tempfile
from configurationHandler.configurations import supportedModelHubs, Model, Dataset

from huggingface_hub import hf_hub_download
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as hfDataset
from datasets import DatasetDict as hfDatasetDict
import kagglehub

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
        self.handlerKaggle.download()
        return "downloaded Models info", "downloaded Datasets info"
    
    def getItems(self):
        """Returns downloaded models and datasets
        """
        pass
            
class DownloadHandler():
    def __init__(self, models: list[Model], datasets: list[Dataset]):
        self.models = models
        self.datasets = datasets
        self.downloadModelFunc = None
        self.downloadDatasetFunc = None
        
    def download(self):
        self.downloadModels()
        self.downloadDatasets() #DATASETS WILL BE ONLY LOCAL WITH DEFINED FORMAT - consult
    
    def downloadModels(self):
        for m in self.models:
            try:
                m.localPath = Path(self.downloadModelFunc(m.uri))
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
    hubName = supportedModelHubs[0]

    def __init__(self, models: list[Model], datasets: list[Dataset]):
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
        # TODO: check with Barbora
        # TODO: add "split" option to cfg - datasets; only if HF
        # TODO: add "split" parameter to configurationHanlder.configurations.Dataset - datasets; only if HF
        # TODO: idk if use "streaming" - consultate or compare performance
        # TODO: hugginface dataset - 
        #       define features - e.g. (image classification: "img", "label"; do it in preprocess script => take X images, preprocess for model, save + create csv with "img, label" tuples (new param in datasetcfg?))
        #       or DATASETS WILL BE ONLY LOCAL WITH DEFINED FORMAT (likey likey)
        ds = hf_load_dataset(uri)
        return ds
        # TODO: move to the handling function along with kaggle dataset - ds = load_dataset(localPath) # hf datasets library
        # print(ds[0]['image'])
        # .select_columns(['image', 'label'])
    
class HandlerKaggle(DownloadHandler):
    hubName = supportedModelHubs[1]

    def __init__(self, models: list[Model], datasets: list[Dataset]):
        super().__init__(models, datasets)
        self.downloadModelFunc = kagglehub.model_download
        self.downloadDatasetFunc = self._downloadDataset

    def _downloadDataset(self, uri: str) -> hfDataset | hfDatasetDict:
        localPath = kagglehub.dataset_download(uri)
        ds = hf_load_dataset(localPath)
        return ds


    #TODO: ds = load_dataset(localPath) # hf datasets library