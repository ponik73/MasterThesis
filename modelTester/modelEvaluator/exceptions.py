from pathlib import Path

class ModelEvaluatorErorr(Exception):
    """Base class for Model Evaluator module errors."""
    pass

class DeviceApiException(ModelEvaluatorErorr):
    """Base class for errors occuring during device API calls."""
    pass

class DeviceApiInitError(DeviceApiException):
    pass

class ModelUploadError(DeviceApiException):
    """Model upload call error."""
    def __init__(self, model_local_path: Path, api_url: str):
        super().__init__(f"Could not upload {model_local_path} to {api_url}")