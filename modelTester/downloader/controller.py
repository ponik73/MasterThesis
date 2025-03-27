from pathlib import Path

import kagglehub
from configurationHandler.schemas import Dataset, Model, SupportedModelFrameworks, SupportedModelHubs
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict as HuggingFaceDatasetDict
from datasets import concatenate_datasets as huggingface_concatenate_datasets
from datasets import load_dataset as huggingface_load_dataset
from huggingface_hub import hf_hub_download as huggingface_hub_download


class DownloaderController:
    def __init__(self, models: list[Model], datasets: list[Dataset]):
        try:
            # Sort models and dataset by platform:
            models_by_hub: dict[SupportedModelHubs, Model] = self.sortByModelHub(models)
            datasets_by_hub: dict[SupportedModelHubs, Dataset] = self.sortByModelHub(datasets)

            # Create handlers for platforms:
            self.handlerHf = HandlerHF(
                models=models_by_hub[SupportedModelHubs.HUGGING_FACE],
                datasets=datasets_by_hub[SupportedModelHubs.HUGGING_FACE],
            )
            self.handlerKaggle = HandlerKaggle(
                models=models_by_hub[SupportedModelHubs.KAGGLE], datasets=datasets_by_hub[SupportedModelHubs.KAGGLE]
            )
        except Exception as ex:
            raise ex

    def sortByModelHub(self, items: list[Model | Dataset]) -> dict[SupportedModelHubs, Model | Dataset]:
        group = {}
        for _, modelHub in SupportedModelHubs.__members__.items():
            group[modelHub] = [x for x in items if x.platform == modelHub]
        return group

    def download(self):
        modelsHf, datasetsHf = self.handlerHf.download()
        modelsKaggle, datasetsKaggle = self.handlerKaggle.download()
        return [*modelsHf, *modelsKaggle], [*datasetsHf, *datasetsKaggle]


class DownloadHandler:
    def __init__(self, models: list[Model], datasets: list[Dataset]):
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
                print(f"{m.name} loaded")
            except Exception:
                m.localPath = None
                print(f'Model "{m.name}" could not be loaded')

    def downloadDatasets(self):
        # If program would consume too much memory, load datasets right before using them
        for d in self.datasets:
            try:
                ds = self.downloadDatasetFunc(d.uri)
                if isinstance(ds, HuggingFaceDatasetDict):
                    ds = huggingface_concatenate_datasets(list(ds.values()))
                if not isinstance(ds, HuggingFaceDataset):
                    raise Exception
                d.dataset = ds
                print(f"{d.name} loaded")
            except Exception:
                print(f'Dataset "{d.name}" could not be loaded')


class HandlerHF(DownloadHandler):
    hubName = SupportedModelHubs.HUGGING_FACE

    def __init__(self, models, datasets):
        super().__init__(models, datasets)
        self.downloadModelFunc = lambda x: huggingface_hub_download(**self._parseModelUri(x))
        self.downloadDatasetFunc = self._downloadDataset

    def _parseModelUri(self, uri: str) -> tuple[str, str, str]:
        namespace, blob = uri.split("/blob/")
        revision, filename = blob.split("/")
        return {"repo_id": namespace, "filename": filename, "revision": revision}

    def _downloadDataset(self, uri: str):
        ds = huggingface_load_dataset(uri, trust_remote_code=True)
        return ds


class HandlerKaggle(DownloadHandler):
    hubName = SupportedModelHubs.KAGGLE

    def __init__(self, models: list[Model], datasets: list[Dataset]):
        super().__init__(models, datasets)
        self.downloadModelFunc = lambda x: [f for f in Path(kagglehub.model_download(x)).iterdir()][0]
        self.downloadDatasetFunc = self._downloadDataset

    def _downloadDataset(self, uri: str) -> HuggingFaceDataset | HuggingFaceDatasetDict:
        localPath = kagglehub.dataset_download(uri)
        ds = huggingface_load_dataset(localPath)
        return ds
