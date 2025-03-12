from requests import Response
from pydantic_core import Url

from ..apiCall import apiCall

@apiCall(
    method="GET",
    endpoint=""
)
def getFingerprintCall(response: Response | None, url: Url, **kwargs) -> str:
    if not response:
        raise
    
    return str(response.json()["fingerprint"]).strip()

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