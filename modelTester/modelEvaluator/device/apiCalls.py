import functools
from typing import Literal
import requests
from requests import Response
from pydantic_core import Url


def apiCall(
        method: Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        endpoint: str       
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(url: Url, **kwargs) -> Response | None:
            try:
                url = Url.build(
                    scheme=url.scheme,
                    host=url.host,
                    port=url.port,
                    path=endpoint
                )
                # Execute request:
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()
                return func(response, url, **kwargs)
            except Exception as e:
                return func(None, url, **kwargs)
        
        return wrapper
    return decorator

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
def evaluateAccuracyTfliteCall(response: Response | None, url: Url, **kwargs) -> bytes:
    if not response:
        raise
    
    return response.content