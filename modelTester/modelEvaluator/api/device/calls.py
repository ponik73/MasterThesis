from requests import Response
from pydantic_core import Url
from typing import Tuple

from ..apiCall import apiCall

@apiCall(
    method="GET",
    endpoint="/"
)
def isServerUpCall(response: Response | None, url: Url, **kwargs) -> bool:
    return response is not None

@apiCall(
    method="GET",
    endpoint="/fingerprint"
)
def getFingerprintCall(response: Response | None, url: Url, **kwargs) -> Tuple[str, str]:
    if not response:
        raise
    
    sys_name = str(response.json()["sys_name"]).strip()
    kernel_version = str(response.json()["kernel_version"]).strip()
    return (sys_name, kernel_version)

@apiCall(
    method="POST",
    endpoint="model/upload"
)
def uploadModelCall(response: Response | None, url: Url, **kwargs):
    pass

@apiCall(
    method="GET",
    endpoint="model/input-info/tflite"
)
def getModelInputInfoTfliteCall(response: Response | None, url: Url, **kwargs):
    if not response:
        raise
    
    return response.json()["inputs"]

@apiCall(
    method="POST",
    endpoint="evaluate/latency/tflite"
)
def evaluateLatencyTfliteCall(response: Response | None, url: Url, **kwargs) -> str:
    if not response:
        raise
    
    return str(response.json()["executableOutput"]).strip()

@apiCall(
    method="POST",
    endpoint="evaluate/accuracy/tflite"
)
def predictTfliteCall(response: Response | None, url: Url, **kwargs) -> bytes:
    if not response:
        raise
    
    return response.content