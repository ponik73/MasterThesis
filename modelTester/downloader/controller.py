from pathlib import Path

import kagglehub
from configurationHandler.schemas import Dataset, Model, SupportedModelFrameworks, SupportedModelHub
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict as HuggingFaceDatasetDict
from datasets import concatenate_datasets as huggingface_concatenate_datasets
from datasets import load_dataset as huggingface_load_dataset
from huggingface_hub import hf_hub_download as huggingface_hub_download


class DownloaderController:
    """Controller for managing the download of models and datasets from supported model hubs like Hugging Face and Kaggle."""

    def __init__(self, models: list[Model], datasets: list[Dataset]) -> None:
        """Initialize the DownloaderController by sorting models and datasets by platform and setting up platform-specific handlers.

        :param models: List of model instances to be downloaded.
        :param datasets: List of dataset instances to be downloaded.
        """
        # Sort models and dataset by platform:
        models_by_hub: dict[SupportedModelHub, Model] = self.sort_by_model_hub(models)
        datasets_by_hub: dict[SupportedModelHub, Dataset] = self.sort_by_model_hub(datasets)

        # Create handlers for platforms:
        self.handlerHf = HandlerHF(
            models=models_by_hub[SupportedModelHub.HUGGING_FACE],
            datasets=datasets_by_hub[SupportedModelHub.HUGGING_FACE],
        )
        self.handlerKaggle = HandlerKaggle(
            models=models_by_hub[SupportedModelHub.KAGGLE], datasets=datasets_by_hub[SupportedModelHub.KAGGLE]
        )

    def sort_by_model_hub(self, items: list[Model | Dataset]) -> dict[SupportedModelHub, Model | Dataset]:
        """Group models or datasets by their associated platform.

        :param items: List of model or dataset instances to group.
        :return: Dictionary mapping each SupportedModelHub to its corresponding list of items.
        """
        group = {}
        for _, model_hub in SupportedModelHub.__members__.items():
            group[model_hub] = [x for x in items if x.platform == model_hub]
        return group

    def download(self) -> tuple[list[Model], list[Dataset]]:
        """Download models and datasets specified in configuration file.

        :return: A tuple containing two lists: downloaded models and datasets.
        """
        models_hf, datasets_hf = self.handlerHf.download()
        models_kaggle, datasets_kaggle = self.handlerKaggle.download()
        return [*models_hf, *models_kaggle], [*datasets_hf, *datasets_kaggle]


class DownloadHandler:
    """Base handler for downloading models and datasets.

    This class is designed to be extended by specific platform handlers like
    HandlerHF and HandlerKaggle, which assign concrete implementations to the
    download functions.
    """

    def __init__(self, models: list[Model], datasets: list[Dataset]) -> None:
        """Initialize the handler with lists of models and datasets.

        :param models: List of model objects to download.
        :param datasets: List of dataset objects to download.
        """
        self.models = models
        self.datasets = datasets
        self.download_model_func = None
        self.download_dataset_func = None

    def download(self) -> tuple[list[Model], list[Dataset]]:
        """Initiate download of both models and datasets."""
        self.download_models()
        self.download_datasets()
        return self.models, self.datasets

    def download_models(self) -> None:
        """Download models using the configured download function.

        Sets `local_path` and `framework` on each model. If a model fails to download,
        its `local_path` is set to None.
        """
        for m in self.models:
            try:
                m.local_path = Path(self.download_model_func(m.uri))
                m.framework = SupportedModelFrameworks(m.local_path.suffix.lstrip("."))
                print(f"{m.name} loaded")
            except Exception:
                m.local_path = None
                print(f'Model "{m.name}" could not be loaded')

    def download_datasets(self) -> None:
        """Download datasets using the configured download function.

        If a dataset is a DatasetDict, it will be merged into a single dataset.
        The resulting dataset is stored in the `dataset` attribute.
        """
        for d in self.datasets:
            try:
                ds = self.download_dataset_func(d.uri)
                if isinstance(ds, HuggingFaceDatasetDict):
                    ds = huggingface_concatenate_datasets(list(ds.values()))
                if not isinstance(ds, HuggingFaceDataset):
                    raise TypeError
                d.dataset = ds
                print(f"{d.name} loaded")
            except Exception:
                print(f'Dataset "{d.name}" could not be loaded')


class HandlerHF(DownloadHandler):
    """Handler for downloading models and datasets from the Hugging Face."""

    def __init__(self, models: list[Model], datasets: list[Dataset]) -> None:
        """Initialize the Hugging Face handler and configure the download functions.

        :param models: List of Hugging Face models.
        :param datasets: List of Hugging Face datasets.
        """
        super().__init__(models, datasets)
        self.download_model_func = lambda x: huggingface_hub_download(**self._parse_model_uri(x))
        self.download_dataset_func = self._download_dataset

    def _parse_model_uri(self, uri: str) -> tuple[str, str, str]:
        namespace, blob = uri.split("/blob/")
        revision, filename = blob.split("/")
        return {"repo_id": namespace, "filename": filename, "revision": revision}

    def _download_dataset(self, uri: str) -> HuggingFaceDataset | HuggingFaceDatasetDict:
        return huggingface_load_dataset(uri, trust_remote_code=True)


class HandlerKaggle(DownloadHandler):
    """Handler for downloading models and datasets from Kaggle."""

    def __init__(self, models: list[Model], datasets: list[Dataset]) -> None:
        """Initialize the Kaggle handler and configure the download functions.

        :param models: List of Kaggle models.
        :param datasets: List of Kaggle datasets.
        """
        super().__init__(models, datasets)
        self.download_model_func = lambda x: next(iter(Path(kagglehub.model_download(x)).iterdir()))
        self.download_dataset_func = self._download_dataset

    def _download_dataset(self, uri: str) -> HuggingFaceDataset | HuggingFaceDatasetDict:
        local_path = kagglehub.dataset_download(uri)
        return huggingface_load_dataset(local_path)
