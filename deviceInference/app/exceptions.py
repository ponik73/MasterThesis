class DeviceApiException(Exception):
    """Base class for runtime exceptions within this API."""
    pass

class ModelNotFoundError(DeviceApiException):
    """Raised when user specified model is not present in model storage."""
    def __init__(self, name:str, framework: str):
        super().__init__(f"Model `{name}.{framework}` not found in model storage.")

class InferenceError(DeviceApiException):
    """Base class for inference exceptions."""
    pass

class TFLiteInferenceError(InferenceError):
    """TFlite inference error."""
    def __init__(self, error_message: str):
        super().__init__(f"TFLite inference failed. Reason: {error_message}")

class LatencyError(DeviceApiException):
    """Base class for latency exceptions."""
    pass

class TFLiteLatencyError(LatencyError):
    def __init__(self, return_code: int, stderr: str):
        super().__init__(f"TFLite latency executable failed. Error code {return_code}; {stderr}")