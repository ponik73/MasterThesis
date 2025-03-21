from datasets import features, formatting
from PIL import Image
from typing import List, Tuple, Any, Dict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from modelEvaluator.api.utils import encodeData, decodeData
from configurationHandler.schemas import Model
from ..accuracyEvaluator import AccuracyEvaluator
from .utils import resizeImages
from user_script.scripts.align_target_predicted import AlignTargetPredictedScript, AlignTargetPredictedFunction

class SingleInputSingleOutputAccuracyEvaluator(AccuracyEvaluator):
    """Class for evaluating single input and single output computer vision models"""
    def __init__(self, model, deviceInterface, dataset):
        super().__init__(model, deviceInterface, dataset)

        # Get the name of the feature which contains images:
        self.imageKey : str | None = None # TODO: specify in config
        self.labelKey : str | None = None # TODO: specify in config
        for key, value in dataset.dataset.features.items():
            if isinstance(value, features.image.Image):
                self.imageKey = key
            if isinstance(value, features.features.ClassLabel):
                self.labelKey = key
        if not self.imageKey or not self.labelKey:
            raise ValueError(f"Dataset {self.dataset.name} does not contain features in required format.")

    def load_align_function(self):
        required_function_name = "align_target_predicted"
        
        script_path = Path("/Users/jakubkasem/Documents-MacBookAir/Uni/DP/MasterThesis/test/scripts/parsing/mobilenetv1_mini_imagenet_classification.py") # TODO: specify config (run specific)
        script = AlignTargetPredictedScript(script_path)
        script.load()
        
        align_function : AlignTargetPredictedFunction | None = script.get_function_by_name(required_function_name)
        if align_function is None:
            raise Exception("TODO")
        return align_function


    def evaluate(self):
        align_function = self.load_align_function()

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
                images = np.array([np.array(x[self.imageKey]) / 255.0 for x in batch]).astype(inputDtype)
                images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
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
            modelOutputs : List[Dict[int, np.typing.NDArray[Any]]] = np.array(decodeData(encodedOutputs))
            
            predictions, targets, num_classes = align_function.call(predictions=modelOutputs, targets=labels)
            if len(self.metrics) == 0:
                self.initMetrics(**{"num_classes": num_classes})
            
            self.updateMetrics(torch.tensor(predictions), torch.tensor(targets))
        
        return self.computeMetrics()
        