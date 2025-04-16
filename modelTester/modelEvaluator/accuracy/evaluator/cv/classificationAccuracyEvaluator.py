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
from .utils import resize_images


class ClassificationAccuracyEvaluator(AccuracyEvaluator):
    """Accuracy evaluator for image classification models."""

    def __init__(
        self, model: Model, device_interface: DeviceInterface, dataset: Dataset, required_script_paths: dict[str, str]
    ) -> None:
        """Initialize the evaluator with the model, device, dataset, and user scripts."""
        super().__init__(model, device_interface, dataset, required_script_paths)

        self.imageKey = dataset.dataset_keys.get("input", None)
        self.labelKey = dataset.dataset_keys.get("target", None)

    def evaluate(self) -> list[MetricResult]:
        """Execute classification accuracy evaluation."""
        # Resize images
        input_shape: list[int] = self.modelInputs[0]["shape"]
        target_size = (input_shape[1], input_shape[2])
        transformed_dataset = resize_images(dataset=self.dataset.dataset, columns={self.imageKey: (target_size, "RGB")})

        # Create data loader and batches:
        input_dtype: str = self.modelInputs[0]["dtype"]

        def _collate_fn(batch) -> dict[str, np.typing.NDArray[Any]]:
            # images = np.array([x[self.imageKey] for x in batch]).astype(input_dtype)

            if np.issubdtype(input_dtype, np.integer):
                images = np.array([x[self.imageKey] for x in batch]).astype(input_dtype)
            elif np.issubdtype(input_dtype, np.floating):
                images = np.array([np.array(x[self.imageKey]) / 255.0 for x in batch]).astype(input_dtype)
                images = np.array([x[self.imageKey] for x in batch]).astype(input_dtype)
            else:
                error_message = "Model input data type is not supported."
                raise TypeError(error_message)

            labels = np.array([x[self.labelKey] for x in batch])
            return {self.imageKey: images, self.labelKey: labels}

        dataloader = DataLoader(
            transformed_dataset, shuffle=True, batch_size=self.BATCH_SIZE, collate_fn=_collate_fn, drop_last=True
        )

        input_idx: int = self.modelInputs[0]["index"]
        # Inference for each batch:
        for batch in dataloader:
            images: np.typing.NDArray[Any] = batch[self.imageKey]
            labels: np.typing.NDArray[Any] = batch[self.labelKey]

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

            predictions, targets, num_classes = self.align_function.call(predictions=model_outputs, targets=labels)
            if len(self.metrics) == 0:
                self.init_metrics(num_classes=num_classes)

            self.update_metrics(torch.tensor(predictions), torch.tensor(targets))

        return self.compute_metrics()
