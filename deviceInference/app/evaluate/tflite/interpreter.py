from pathlib import Path
from typing import Any

import numpy as np
import tflite_runtime.interpreter as tflite
from exceptions import TFLiteInferenceError

# from tensorflow import lite as tflite
from ..schemas import Batch


class TfliteInterpreter:
    """Class for inference of tflite models encapsulating tflite interpreter."""

    def __init__(self) -> None:
        """TfliteInterpreter constructor."""
        self.model_path: Path | None = None
        self.interpreter: tflite.Interpreter | None = None
        self.detailsInput = None
        self.detailsOutput = None

    def start(self, model_path: Path) -> None:
        """Initialize tflite interpreter.

        :param model_path: path of model that will be infered
        """
        # Return if interpreter is running:
        if self.interpreter and self.model_path == model_path:
            return

        # Init interpreter:
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path.as_posix())  # ,
            # experimental_delegates=ext_delegate)
            self.detailsInput: list[dict[str, Any]] = self.interpreter.get_input_details()
            self.detailsOutput = self.interpreter.get_output_details()
            self.model_path = model_path
        except Exception as e:
            error_message = "Unable to initiate TFLite interpreter."
            raise TFLiteInferenceError(error_message) from e

    def get_input_details(self) -> list[dict[str, Any]]:
        """Get input details of model set for inference.

        :return: List of dictionaries containing node details
        """
        if not self.detailsInput:
            error_message = "TFLite interpreter has not returned input details."
            raise TFLiteInferenceError(error_message)

        return self.detailsInput

    def get_output_indexes(self) -> list[int]:
        """Get indexes of output node(s).

        :return: indexes of output node(s)
        """
        if not self.detailsOutput:
            error_message = "TFLite interpreter has not returned output details."
            raise TFLiteInferenceError(error_message)

        return [x["index"] for x in self.detailsOutput]

    def inference(self, batch: Batch) -> list[dict[int, Any]]:
        """Execute inference of tflite model over a batch.

        :param batch: model input batch
        :return: List of dictionaries that contain output node value mapped to an index.
        """
        try:
            self.interpreter.allocate_tensors()

            results: list[dict[int, Any]] = []  # Batch results
            # Execute the inference of batch:
            for item in batch.items:
                # Fed data to each input:
                for input_idx, data in item.items():
                    self.interpreter.set_tensor(input_idx, np.array([data]))
                # Inference:
                self.interpreter.invoke()
                # Retrieve inference result:
                res = {}  # Results of one item in the batch
                for output_idx in self.get_output_indexes():
                    res[output_idx] = self.interpreter.get_tensor(output_idx)
                # Add the results of item to the batch result list:
                results.append(res)

        except Exception as e:
            raise TFLiteInferenceError(str(e)) from e

        return results


tflite_interpreter = TfliteInterpreter()
