from typing import Any

from modelEvaluator.exceptions import AccuracyAssessmentError, FingerprintError, LatencyAssessmentError, ModelInfoError
from pydantic_core import Url
from requests import Response

from ..apiCall import api_call


@api_call(method="GET", endpoint="/")
def is_server_up_call(response: Response | None, url: Url, **kwargs: dict[str, Any]) -> bool:
    """Ping device API "alive" endpoint."""
    return response is not None


@api_call(method="GET", endpoint="/fingerprint")
def get_fingerprint_call(response: Response | None, url: Url, **kwargs: dict[str, Any]) -> tuple[str, str]:
    """Call fingerprint endpoint on the device API."""
    if not response:
        raise FingerprintError(str(url))

    sys_name = str(response.json()["sys_name"]).strip()
    kernel_version = str(response.json()["kernel_version"]).strip()

    if sys_name is None or kernel_version is None:
        raise FingerprintError(str(url))

    return (sys_name, kernel_version)


@api_call(method="POST", endpoint="model/upload")
def upload_model_call(response: Response | None, url: Url, **kwargs: dict[str, Any]) -> None:
    """Call model upload endpoint on the device API."""


@api_call(method="GET", endpoint="model/input-info/tflite")
def get_model_input_info_tflite_call(response: Response | None, url: Url, **kwargs: dict[str, Any]) -> list[dict]:
    """Call tflite model info endpoint on the device API."""
    if not response:
        raise ModelInfoError

    return response.json()["inputs"]


@api_call(method="POST", endpoint="evaluate/latency/tflite")
def evaluate_latency_tflite_call(response: Response | None, url: Url, **kwargs: dict[str, Any]) -> str:
    """Call tflite latency endpoint on the device API."""
    if not response:
        raise LatencyAssessmentError

    return str(response.json()["executable_output"]).strip()


@api_call(method="POST", endpoint="evaluate/accuracy/tflite")
def predict_tflite_call(response: Response | None, url: Url, **kwargs: dict[str, Any]) -> bytes:
    """Call tflite accuracy endpoint on the device API."""
    if not response:
        raise AccuracyAssessmentError

    return response.content
