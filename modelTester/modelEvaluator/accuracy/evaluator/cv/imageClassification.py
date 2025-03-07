from datasets import features, formatting
from PIL import Image
from typing import List, Tuple, Any, Dict
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

from modelEvaluator.device.utils import encodeData, decodeData
from configurationHandler.configurations import Model
from ..accuracyEvaluator import AccuracyEvaluator
from .utils import resizeImages
from modelEvaluator.accuracy.metrics.metric import EvaluationMetrics
from modelEvaluator.accuracy.metrics.classification import AccuracyMacro, AccuracyMicro, RecallMacro, RecallMicro

class ImageClassificationAccuracyEvaluator(AccuracyEvaluator):
    def __init__(self, model, deviceInterface, dataset):
        super().__init__(model, deviceInterface, dataset)

        # Get the name of the feature which contains images:
        self.imageKey : str | None = None
        self.labelKey : str | None = None
        for key, value in dataset.dataset.features.items():
            if isinstance(value, features.image.Image):
                self.imageKey = key
            if isinstance(value, features.features.ClassLabel):
                self.labelKey = key
        if not self.imageKey or not self.labelKey:
            raise ValueError(f"Dataset {self.dataset.name} does not contain features in required format.")


    def _metrics(self):
        return [
            AccuracyMacro(), AccuracyMicro(), RecallMacro(), RecallMicro()
        ]


    def _predict(self):
        # self.targets = np.load("target.npy")
        # self.predictions = np.load("preds.npy", allow_pickle=True)
        # return
        # Resize images
        inputShape : List[int] = self.modelInputs[0]["shape"]
        targetSize = (inputShape[1], inputShape[2])
        transformedDataset = resizeImages(
            dataset=self.dataset.dataset,
            columns={
                self.imageKey: (targetSize, "RGB")
            }
        )

        # Create data loader and batches:
        inputDtype : str = self.modelInputs[0]["dtype"]
        def collate_fn(batch):
            # images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            
            if np.issubdtype(inputDtype, np.integer):
                images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            elif np.issubdtype(inputDtype, np.floating):
                images = np.array([x[self.imageKey] / 255.0 for x in batch]).astype(inputDtype)
            else:
                raise ValueError("TODO")
            
            labels = np.array([x[self.labelKey] for x in batch])
            return {
                self.imageKey: images,
                self.labelKey: labels
            }
        dataloader = DataLoader(transformedDataset, shuffle=True, batch_size=self.BATCH_SIZE, collate_fn=collate_fn, drop_last=True)
        
        inputIdx : int = self.modelInputs[0]["index"]
        # Inference for each batch:
        for batch in dataloader:
            images : np.typing.NDArray[Any] = batch[self.imageKey]
            labels : np.typing.NDArray[Any] = batch[self.labelKey]
            
            if self.targets is None:
                self.targets = np.array(labels)
            else:
                # self.targets = np.concatenate((self.targets, [labels]))
                self.targets = np.append(self.targets, labels, axis=0)
            
            mappingInputData : List[Dict[int, np.typing.NDArray[Any]]] = [{inputIdx: x} for x in images]

            # Inference:
            encodedOutputs = self.deviceInterface.predict(
                modelCustomName=self.model.name,
                modelFramework=self.model.framework,
                encodedBatch=encodeData(mappingInputData)
            )
            if not encodedOutputs:
                raise Exception(f'[{self.deviceInterface.name}] Error during on device inference.')
            
            # Decode inference result and append it to the predictions:
            modelOutputs : List[Dict[int, np.typing.NDArray[Any]]] = decodeData(encodedOutputs)
            if self.predictions is None:
                self.predictions = np.array(modelOutputs)
            else:
                self.predictions = np.append(self.predictions, modelOutputs, axis=0)
    
    
    def _calculateMetrics(self):
        # TODO: model-dataset label mapping?

        # Single output index:
        outputIdx = list(self.predictions.item(0).keys())[0]
        # Exctract the output values:
        self.predictions = np.array([x[outputIdx] for x in self.predictions])
        # Argmax:
        self.predictions = np.array([x.argmax().item() for x in self.predictions])


        self.predictions = np.array([x - 1 for x in self.predictions]) # Dataset has 1000 classes, model has 1001 outpus (index 0 is for background)

        return EvaluationMetrics().compute(metrics=self._metrics(), predictions=self.predictions, targets=self.targets)

    def _task(self):
        return Model.SupportedTasks.IMAGE_CLASSIFICATION