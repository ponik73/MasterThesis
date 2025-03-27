from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from modelEvaluator.api.utils import decodeData, encodeData

from ..accuracyEvaluator import AccuracyEvaluator
from .utils import resizeImages


class SingleInputSingleOutputAccuracyEvaluator(AccuracyEvaluator):
    """Class for evaluating single input and single output computer vision models"""

    def __init__(self, model, deviceInterface, dataset, required_script_paths):
        super().__init__(model, deviceInterface, dataset, required_script_paths)

        self.imageKey = dataset.dataset_keys.get("input", None)
        self.labelKey = dataset.dataset_keys.get("target", None)

    def evaluate(self):
        # Resize images
        inputShape: list[int] = self.modelInputs[0]["shape"]
        targetSize = (inputShape[1], inputShape[2])
        transformedDataset = resizeImages(dataset=self.dataset.dataset, columns={self.imageKey: (targetSize, "RGB")})

        # Create data loader and batches:
        inputDtype: str = self.modelInputs[0]["dtype"]

        def collate_fn(batch):
            # images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)

            if np.issubdtype(inputDtype, np.integer):
                images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            elif np.issubdtype(inputDtype, np.floating):
                images = np.array([np.array(x[self.imageKey]) / 255.0 for x in batch]).astype(inputDtype)
                images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            else:
                raise TypeError("Model input data type is not supported.")

            labels = np.array([x[self.labelKey] for x in batch])
            return {self.imageKey: images, self.labelKey: labels}

        dataloader = DataLoader(
            transformedDataset, shuffle=True, batch_size=self.BATCH_SIZE, collate_fn=collate_fn, drop_last=True
        )

        inputIdx: int = self.modelInputs[0]["index"]
        # Inference for each batch:
        for batch in dataloader:
            images: np.typing.NDArray[Any] = batch[self.imageKey]
            labels: np.typing.NDArray[Any] = batch[self.labelKey]

            mappingInputData: list[dict[int, np.typing.NDArray[Any]]] = [{inputIdx: x} for x in images]

            # Inference:
            encodedOutputs = self.deviceInterface.predict(
                model_custom_name=self.model.name,
                modelFramework=self.model.framework,
                encoded_batch=encodeData(mappingInputData),
            )
            if not encodedOutputs:
                raise Exception(f"[{self.deviceInterface.name}] Error during on device inference.")

            # Decode inference result and append it to the predictions:
            model_outputs: list[dict[int, np.typing.NDArray[Any]]] = np.array(decodeData(encodedOutputs))

            predictions, targets, num_classes = self.align_function.call(predictions=model_outputs, targets=labels)
            if len(self.metrics) == 0:
                self.initMetrics(num_classes=num_classes)

            self.updateMetrics(torch.tensor(predictions), torch.tensor(targets))

        return self.computeMetrics()
