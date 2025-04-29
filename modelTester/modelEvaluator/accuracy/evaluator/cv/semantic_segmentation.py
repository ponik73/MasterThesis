from typing import Any

import numpy as np
import torch
from configurationHandler.schemas import Dataset, Model
from modelEvaluator.accuracy.metrics.metric import MetricResult
from modelEvaluator.api.device.interface import DeviceInterface
from modelEvaluator.api.utils import decode_data, encode_data
from modelEvaluator.exceptions import OndeviceInferenceError
from torch.utils.data import DataLoader

from ..accuracyEvaluator import AccuracyEvaluator
from .utils import resize_images, nested_read


class SemanticSegmentationAccuracyEvaluator(AccuracyEvaluator):
    """Evaluator for semantic segmentation models.

    This class resizes input images and annotations, runs the model on-device for evaluation,
    aligns predictions and targets, and computes defined metrics for semantic segmentation tasks.
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
        self.annotationKey = dataset.dataset_keys.get("target", None)

    def evaluate(self) -> list[MetricResult]:
        """Execute the semantic segmentation inference and metric evaluation."""
        # Resize images
        input_shape: list[int] = self.modelInputs[0]["shape"]
        target_size = (input_shape[1], input_shape[2])
        transformed_dataset = resize_images(
            dataset=self.dataset.dataset,
            columns=[
                (self.imageKey, (target_size, "RGB")),
                (self.annotationKey, (target_size, "P")),
            ],
        )

        # Create data loader and batches:
        input_dtype: str = self.modelInputs[0]["dtype"]

        def _collate_fn(batch) -> dict[str, np.typing.NDArray[Any]]:
            if np.issubdtype(input_dtype, np.integer):
                # images = np.array([x[self.imageKey] for x in batch]).astype(input_dtype)
                images = np.array([nested_read(x, self.imageKey) for x in batch]).astype(input_dtype)
            elif np.issubdtype(input_dtype, np.floating):
                images = np.array([np.array(nested_read(x, self.imageKey)) / 255.0 for x in batch]).astype(input_dtype)
                # images = np.array([np.array(x[self.imageKey]) / 255.0 for x in batch]).astype(input_dtype)
                # images = np.array([x[self.imageKey] for x in batch]).astype(input_dtype)
            else:
                error_message = "Model input data type is not supported."
                raise TypeError(error_message)

            # annotations = np.array([x[self.annotationKey] for x in batch])
            annotations = np.array([nested_read(x, self.annotationKey) for x in batch])
            return images, annotations

        dataloader = DataLoader(
            transformed_dataset, shuffle=True, batch_size=self.BATCH_SIZE, collate_fn=_collate_fn, drop_last=True
        )

        input_idx: int = self.modelInputs[0]["index"]
        # Inference for each batch:
        for images, annotations in dataloader:
            mapping_input_data: list[dict[int, np.typing.NDArray[Any]]] = [{input_idx: x} for x in images]
            # Inference:
            encoded_outputs = self.device_interface.predict(
                model_custom_name=self.model.name,
                model_framework=self.model.framework,
                encoded_batch=encode_data(mapping_input_data),
            )
            if not encoded_outputs:
                raise OndeviceInferenceError(model_local_path=self.model.local_path, api_url=self.device_interface.url)

            # Decode inference result and append it to the predictions:
            model_outputs: list[dict[int, np.typing.NDArray[Any]]] = np.array(decode_data(encoded_outputs))

            predictions, targets, num_classes, input_format = self.align_function.call(
                predictions=model_outputs, targets=annotations
            )
            if len(self.metrics) == 0:
                self.init_metrics(num_classes=num_classes, input_format=input_format)

            self.update_metrics(torch.tensor(predictions), torch.tensor(targets))

            break
        return self.compute_metrics()
