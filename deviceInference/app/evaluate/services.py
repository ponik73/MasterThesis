import codecs
import pickle
from pathlib import Path
from typing import Any, Literal

from exceptions import EncodingDecodingError, FrameworkNotSupportedError

from .schemas import Batch
from .tflite.services import accuracy_assessment_tflite, latency_assessment_tflite


def latency_assessment(framework_nn: Literal[".tflite"], model_path: Path, temp_dir_path: Path) -> str:
    """Choose and call the latency assessment function.

    :param framework_nn: neural network framework
    :param model_path: model path
    :param temp_dir_path: temporary directory path
    :return: Corresponding latency assessment function output
    """
    latency_assessment_functions = {".tflite": latency_assessment_tflite}

    if framework_nn not in latency_assessment_functions:
        raise FrameworkNotSupportedError(framework=framework_nn)

    return latency_assessment_functions[framework_nn](model_path, temp_dir_path)


def accuracy_assessment(framework_nn: Literal[".tflite"], model_path: Path, encoded_batch: bytes) -> bytes:
    """Choose and call the accuracy assessment function.

    :param framework_nn: neural network framework
    :param model_path: model path
    :param encoded_batch: model input batch encoded as bytes
    :return: Corresponding accuracy assessment function output
    """
    assessment_functions = {".tflite": accuracy_assessment_tflite}

    # Decode batch data:
    decoded_batch = decode_batch(encoded_batch)

    if framework_nn not in assessment_functions:
        raise FrameworkNotSupportedError(framework=framework_nn)

    # Execute inference:
    inference_result = assessment_functions[framework_nn](model_path, decoded_batch)

    return encode_model_outputs(inference_result)


def decode_batch(encoded_batch: bytes) -> Batch:
    """Decode batch encoded as bytes.

    :param encoded_batch: batch encoded as bytes
    :return: decoded batch
    """
    try:
        return Batch(items=pickle.loads(codecs.decode(encoded_batch, "base64")))
    except ValueError as e:
        raise EncodingDecodingError from e


def encode_model_outputs(model_outputs: list[dict[int, Any]]) -> bytes:
    """Encode model outpus as bytes.

    :param model_outputs: outputs of the inference on a model
    :return: Outputs encoded as bytes
    """
    return codecs.encode(pickle.dumps(model_outputs, protocol=pickle.HIGHEST_PROTOCOL), "base64")
