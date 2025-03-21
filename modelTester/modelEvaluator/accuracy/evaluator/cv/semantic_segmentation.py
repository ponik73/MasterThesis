from datasets import features, formatting
from PIL import Image
from typing import List, Tuple, Any, Dict
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import torch

from modelEvaluator.api.utils import encodeData, decodeData
from configurationHandler.schemas import Model
from ..accuracyEvaluator import AccuracyEvaluator
from .utils import resizeImages
from user_script.scripts.align_target_predicted import AlignTargetPredictedScript, AlignTargetPredictedFunction

class SemanticSegmentationAccuracyEvaluator(AccuracyEvaluator):
    def __init__(self, model, deviceInterface, dataset, required_script_paths):
        super().__init__(model, deviceInterface, dataset, required_script_paths)

        self.imageKey = dataset.dataset_keys.get("input", None)
        self.annotationKey = dataset.dataset_keys.get("target", None)

    def load_align_function(self):
        required_script_name = "align_target_predicted"
        required_function_name = "align_target_predicted"
        
        script_path = Path(self.required_script_paths.get(required_script_name, None))
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
                self.imageKey: (targetSize, "RGB"),
                self.annotationKey: (targetSize, "P"),
            }
        )

        # Create data loader and batches:
        inputDtype : str = self.modelInputs[0]["dtype"]
        def collate_fn(batch):            
            if np.issubdtype(inputDtype, np.integer):
                images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            elif np.issubdtype(inputDtype, np.floating):
                images = np.array([np.array(x[self.imageKey]) / 255.0 for x in batch]).astype(inputDtype)
                # images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            else:
                raise ValueError("TODO")
            
            
            annotations = np.array([x[self.annotationKey] for x in batch])
            return {
                self.imageKey: images,
                self.annotationKey: annotations
            }
        dataloader = DataLoader(transformedDataset, shuffle=True, batch_size=self.BATCH_SIZE, collate_fn=collate_fn, drop_last=True)
        
        inputIdx : int = self.modelInputs[0]["index"]
        # Inference for each batch:
        for batch in dataloader:
            images : np.typing.NDArray[Any] = batch[self.imageKey]
            annotations : np.typing.NDArray[Any] = batch[self.annotationKey]
            
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

            predictions, targets, num_classes, input_format = align_function.call(predictions=modelOutputs, targets=annotations)
            if len(self.metrics) == 0:
                self.initMetrics(
                    **{
                        "num_classes": num_classes,
                        "input_format": input_format
                    }
                )
            
            self.updateMetrics(torch.tensor(predictions), torch.tensor(targets))
            
            break
        return self.computeMetrics()
     