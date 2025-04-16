from typing import Dict, List, Any
from numpy.typing import NDArray
import numpy as np
from torchmetrics.functional.detection.iou import intersection_over_union
from torchmetrics.detection.iou import IntersectionOverUnion
from torch.utils.data import DataLoader
import torch

from ..accuracyEvaluator import AccuracyEvaluator
from .utils import resizeImages, resize_bboxes, nested_read
from modelEvaluator.api.utils import encodeData, decodeData

class ObjectDetectionAccuracyEvaluator(AccuracyEvaluator):
    """Class for evaluating single input and single output computer vision models"""
    
    def __init__(self, model, deviceInterface, dataset, required_script_paths):
        super().__init__(model, deviceInterface, dataset, required_script_paths)

        self.imageKey = dataset.dataset_keys.get("input", None)
        self.bboxKey = ['objects', 'bbox'] # TODO: modify config (all the keys will be lists/tuples)
        self.categoriesKey = ['objects', 'categories'] # TODO modify config (all the keys will be lists/tuples)

    def evaluate(self):
        inputShape : List[int] = self.modelInputs[0]["shape"]
        targetSize = (inputShape[1], inputShape[2])

        # Resize bboxes:
        transformedDataset = resize_bboxes(
            dataset=self.dataset.dataset,
            targetSize=targetSize,
            imgKey=self.imageKey,
            bboxKey=self.bboxKey
        )

        # Resize images:
        transformedDataset = resizeImages(
            dataset=transformedDataset,
            columns={
                self.imageKey: (targetSize, "RGB")
            }
        )

        # Create data loader and batches:
        inputDtype : str = self.modelInputs[0]["dtype"]
        def collate_fn(batch):            
            if np.issubdtype(inputDtype, np.integer):
                images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            elif np.issubdtype(inputDtype, np.floating):
                images = np.array([np.array(x[self.imageKey]) / 255.0 for x in batch]).astype(inputDtype)
                images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            else:
                raise TypeError("Model input data type is not supported.")
            
            bboxes = [nested_read(x, self.bboxKey) for x in batch]
            categories = [nested_read(x, self.categoriesKey) for x in batch]
            return images, bboxes, categories

        dataloader = DataLoader(transformedDataset, shuffle=True, batch_size=self.BATCH_SIZE, collate_fn=collate_fn, drop_last=True)
        inputIdx : int = self.modelInputs[0]["index"]
        
        # Inference for each batch:
        for images, bboxes, categories in dataloader:
            
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

            predictions, targets, box_format = self.align_function.call(
                predictions=modelOutputs,
                targets=(bboxes, categories)
            )

            empty = []
            _predictions = []
            _targets = []
            for i, x in enumerate(predictions):
                if len(x["boxes"]) == 0:
                    empty.append(i)
                    continue
                x["boxes"] = torch.tensor(x["boxes"])
                x["scores"] = torch.tensor(x["scores"])
                x["labels"] = torch.tensor(x["labels"])
                _predictions.append(x)
            for i, x in enumerate(targets):
                if i in empty:
                    continue
                x["boxes"] = torch.tensor(x["boxes"])
                x["labels"] = torch.tensor(x["labels"])
                _targets.append(x)

            if len(self.metrics) == 0:
                self.initMetrics(**{"box_format": box_format})
            
            self.updateMetrics(_predictions, _targets)
        
        return self.computeMetrics()

        