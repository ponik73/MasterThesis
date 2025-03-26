class DeviceApiException(Exception):
    """Base class for runtime exceptions within this API."""
    pass

class ModelUploadError(DeviceApiException):
    """Erorr raised during model upload process."""
    pass

class ModelNotFoundError(DeviceApiException):
    """Raised when user specified model is not present in model storage."""
    def __init__(self, name:str, framework: str):
        super().__init__(f"Model `{name}.{framework}` not found in model storage.")

class FrameworkNotSupportedError(DeviceApiException):
    """Model of not supported framework."""
    def __init__(self, framework: str):
        super().__init__(f"Framework `{framework}` is not supported.")

class ArchitectureDeterminationError(DeviceApiException):
    """Could not determine architecture of a device."""
    def __init__(self):
        super().__init__("Could not determine device architecture.")

class ArchitectureNotSupportedError(DeviceApiException):
    """Device architecture is not supported."""
    def __init__(self, arch: str):
        super().__init__(f"Architecture `{arch}` is not supported.")

class EncodingDecodingError(DeviceApiException):
    """Error during data encoding or decoding."""
    pass

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

class TFLiteLatencyExecutableDownloadError(LatencyError):
    """Executable could not be downloaded."""
    def __init__(self, url: str):
        super().__init__(f"TFLite executable could not be downloded from {url}.")

class TFLiteLatencyError(LatencyError):
    def __init__(self, return_code: int, stderr: str):
        super().__init__(f"TFLite latency executable failed. Error code {return_code}; {stderr}")

class FingerprintError(DeviceApiException):
    """Error raised during fingerprint retrieval."""
    pass