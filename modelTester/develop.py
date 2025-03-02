from modelEvaluator.pipeline import RunPipeline, DeviceInterface
from pydantic_core import Url
from datasets import load_dataset as hf_load_dataset
import kagglehub
from typing import List, Tuple
from configurationHandler.configurations import Model, Dataset
from pathlib import Path

def loadDs(path: str):
    # localPath = kagglehub.dataset_download(path)
    localPath = '/Users/jakubkasem/.cache/kagglehub/datasets/samrat230599/fastai-imagenet/versions/3'
    ds = hf_load_dataset(localPath)
    ds = ds[next(iter(ds))]
    return ds

def setupPipeline():
    deviceInterface = DeviceInterface(
        url=Url("http://localhost:8000"),
        name="localhostik"
    )

    model = Model(
        name="mobilenet",
        task="image-classification",
        input_shape=[1,1,1],
        platform="hf",
        uri="nejaka/uri"
    )
    model.localPath = Path("/Users/jakubkasem/Downloads/mobilenet_v1_1.0_224_quant.tflite")

    dataset = Dataset(
        name="datasetik",
        platform="kaggle",
        uri="nejaka/uri"
    )
    dataset.dataset = loadDs("samrat230599/fastai-imagenet")

    pairModelDataset : Tuple[Model, Dataset] = (model, dataset)

    return RunPipeline(
        deviceInterface=deviceInterface,
        pairsModelDataset=[pairModelDataset]
    )
    

if __name__ == "__main__":
    pipeline = setupPipeline()
    pipeline.execute()

    exit(0)