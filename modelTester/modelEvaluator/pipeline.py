from typing import List, Tuple, Any, Dict
from pathlib import Path
# from datasets import Dataset as hfDataset
from datasets import features, formatting
from PIL import Image
import numpy as np
import pickle
import codecs
from sys import getsizeof
from torch.utils.data import DataLoader

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
        
        # Framework of the model derived from file extension:
        modelFramework = model.localPath.suffix.lstrip(".")

        # Evaluate latency on device:
        latencyRawOutput = self.deviceInterface.evaluateLatency(model.name, modelFramework)
        # Parse the output:
        if not latencyRawOutput:
            raise Exception(f'[{self.deviceInterface.name}] Error during latency evaluation.')
        
        parser = latencyParserFactory(modelFramework)
        latencyResults = parser.parse(latencyRawOutput)
    
        # # Evaluate accuracy on device:
        SEED = 690
        np.random.seed(SEED)
        BATCH_SIZE = 64 # TODO: how big batch?

        # TODO: this implementation works for computer vision, make it separate class
        # Get the name of the feature which contains images:
        imageKey : str | None = None # TODO: maybe specify in config
        labelKey : str | None = None # TODO: maybe specify in config
        # TODO: area key ?? segmentation + detection
        for key, value in dataset.dataset.features.items():
            if isinstance(value, features.image.Image):
                imageKey = key
            if isinstance(value, features.features.ClassLabel):
                labelKey = key

        # Get model input size from the API:
        modelInputs = self.deviceInterface.getModelInputInfo(
            modelCustomName=model.name,
            modelFramework=modelFramework
        )
        # TODO: multiple inputs??
        inputIdx : int = modelInputs[0]["index"]
        inputShape : List[int] = modelInputs[0]["shape"]
        inputDtype : str = modelInputs[0]["dtype"]
        targetSize = (inputShape[1], inputShape[2])
        
        # Resize images within dataset:
        def transforms(item: formatting.formatting.LazyDict):
            image : Image.Image = item[imageKey]
            item[imageKey] = image.convert("RGB").resize(targetSize, resample=Image.Resampling.LANCZOS)
            return item
        transformedDataset = dataset.dataset.map(transforms)

        # Create data loader and batches:
        def collate_fn(batch):
            images = np.array([x[imageKey] for x in batch]).astype(inputDtype)
            labels = np.array([x[labelKey] for x in batch])
            return {
                imageKey: images,
                labelKey: labels
            }
        dataloader = DataLoader(transformedDataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        
        modelOutputsOverall : List[Tuple[Any, Any]] = []
        # Inference for each batch:
        for batch in dataloader:
            images : np.typing.NDArray[Any] = batch[imageKey]
            labels : np.typing.NDArray[Any] = batch[labelKey] # TODO: multiple ground truths; area key ?? segmentation + detection
            
            # TODO: multiple inputs??
            mappingInputData : List[Dict[int, np.typing.NDArray[Any]]] = [{inputIdx: x} for x in images]

            # Inference:
            encodedOutputs = self.deviceInterface.evaluateAccuracy(
                modelCustomName=model.name,
                modelFramework=modelFramework,
                encodedBatch=encodeBatch(mappingInputData)
            )
            if not encodedOutputs:
                raise Exception(f'[{self.deviceInterface.name}] Error during on device inference.')
            modelOutputs : List[Dict[int, np.typing.NDArray[Any]]] = decodeModelOutputs(encodedOutputs)
            modelOutputsOverall.extend(list(zip(modelOutputs, labels)))
            if len(modelOutputsOverall) > 5000: # Max number of samples??
                break
            
        # TODO: Calculate metrics:
        # Model-dataset label mapping: TODO cfg
        # labelMapping = {
        #     # Model : Dataset
        #     483: 0, # cassette_player
        #     492: 1, # chain_saw
        #     498: 2, # church
        #     218: 3,# english_springer
        #     567: 4,# french_horn
        #     570: 5,# garbage_truck
        #     572: 6,# gas_pump
        #     575: 7,# golf_ball
        #     702: 8,# parachute
        #     1: 9# tench,
        # }
        # correct = 0
        # for prediction, groundTruth in modelOutputsOverall:
        #     outputClass = prediction[87]
        #     outputClassValue = outputClass.argmax().item()
            
        #     if labelMapping.get(outputClassValue, -1) == groundTruth:
        #         correct += 1
        # print(f"correct/all = {correct}/{len(modelOutputsOverall)}")
        pass
        
        # TODO: Write results to database: (TODO maybe new fastapi for db)
        pass
        
    
def encodeBatch(batch: Any):
    return codecs.encode(pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL), "base64")

def decodeModelOutputs(outputs: bytes):
    try:
        return np.array(pickle.loads(codecs.decode(outputs, "base64")))
    except Exception as e:
        raise Exception(f"Error during decoding model outputs. Reason: '{e}'.") from e