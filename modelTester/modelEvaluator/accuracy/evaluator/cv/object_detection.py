from typing import Any

import numpy as np
import torch
from configurationHandler.schemas import Dataset, Model
from modelEvaluator.api.device.interface import DeviceInterface
from modelEvaluator.accuracy.metrics.metric import MetricResult
from modelEvaluator.api.utils import decodeData, encodeData
from torch.utils.data import DataLoader

from ..accuracyEvaluator import AccuracyEvaluator
from .utils import nested_read, resize_bboxes, resizeImages
from modelEvaluator.exceptions import OndeviceInferenceError


class ObjectDetectionAccuracyEvaluator(AccuracyEvaluator):
    """Evaluator for object detection models.

    This class resizes input images and ground truth bounding boxes, runs the model on-device for evaluation,
    aligns predictions and targets, and computes defined metrics for object detection tasks.
    """

    def __init__(
        self, model: Model, device_interface: DeviceInterface, dataset: Dataset, required_script_paths: dict[str, str]
    ) -> None:
        """Create SemanticSegmentationAccuracyEvaluator instance.

        :param model: The model to be evaluated.
        :param device_interface: Interface to communicate with the device that executes the model.
        :param dataset: The dataset used for evaluation, with expected `input` and `target` keys in `dataset_keys`.
        :param required_script_paths: Dictionary mapping required script names to file paths.
        """
        super().__init__(model, device_interface, dataset, required_script_paths)

        self.imageKey = dataset.dataset_keys.get("input", None)
        self.bbox_key = ["objects", "bbox"]  # TODO: modify config (all the keys will be lists/tuples)
        self.categoriesKey = ["objects", "categories"]  # TODO modify config (all the keys will be lists/tuples)

    def evaluate(self) -> list[MetricResult]:
        """Execute the object detection inference and metric evaluation."""
        input_shape: list[int] = self.modelInputs[0]["shape"]
        target_size = (input_shape[1], input_shape[2])

        # Resize bboxes:
        transformed_dataset = resize_bboxes(
            dataset=self.dataset.dataset, target_size=target_size, img_key=self.imageKey, bbox_key=self.bbox_key
        )

        # Resize images:
        transformed_dataset = resizeImages(dataset=transformed_dataset, columns={self.imageKey: (target_size, "RGB")})

        # Create data loader and batches:
        input_dtype: str = self.modelInputs[0]["dtype"]

        def _collate_fn(batch) -> tuple[np.typing.NDArray[Any], list, list]:
            if np.issubdtype(input_dtype, np.integer):
                images = np.array([x[self.imageKey] for x in batch]).astype(input_dtype)
            elif np.issubdtype(input_dtype, np.floating):
                images = np.array([np.array(x[self.imageKey]) / 255.0 for x in batch]).astype(input_dtype)
                images = np.array([x[self.imageKey] for x in batch]).astype(input_dtype)
            else:
                error_message = "Model input data type is not supported."
                raise TypeError(error_message)

            bboxes = [nested_read(x, self.bbox_key) for x in batch]
            categories = [nested_read(x, self.categoriesKey) for x in batch]
            return images, bboxes, categories

        dataloader = DataLoader(
            transformed_dataset, shuffle=True, batch_size=self.BATCH_SIZE, collate_fn=_collate_fn, drop_last=True
        )
        input_idx: int = self.modelInputs[0]["index"]

        # Inference for each batch:
        for images, bboxes, categories in dataloader:
            mapping_input_data: list[dict[int, np.typing.NDArray[Any]]] = [{input_idx: x} for x in images]

            # Inference:
            encoded_outputs = self.device_interface.predict(
                modelCustomName=self.model.name,
                modelFramework=self.model.framework,
                encodedBatch=encodeData(mapping_input_data),
            )
            if not encoded_outputs:
                raise OndeviceInferenceError(model_local_path=self.model.local_path, api_url=self.device_interface.url)

            # Decode inference result and append it to the predictions:
            model_outputs: list[dict[int, np.typing.NDArray[Any]]] = np.array(decodeData(encoded_outputs))

            predictions, targets, box_format = self.align_function.call(
                predictions=model_outputs, targets=(bboxes, categories)
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
                self.init_metrics(box_format=box_format)

            self.update_metrics(_predictions, _targets)

        return self.compute_metrics()
