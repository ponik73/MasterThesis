class DeviceApiError(Exception):
    """Base class for runtime exceptions within this API."""


class ModelUploadError(DeviceApiError):
    """Erorr raised during model upload process."""


class ModelNotFoundError(DeviceApiError):
    """Raised when user specified model is not present in model storage."""

    def __init__(self, name: str, framework: str) -> None:
        """ModelNotFoundError constructor.

        :param name: name of the model
        :param framework: model's framework
        """
        super().__init__(f"Model `{name}.{framework}` not found in model storage.")


class FrameworkNotSupportedError(DeviceApiError):
    """Model of not supported framework."""

    def __init__(self, framework: str) -> None:
        """FrameworkNotSupportedError constructor."""
        super().__init__(f"Framework `{framework}` is not supported.")


class ArchitectureDeterminationError(DeviceApiError):
    """Could not determine architecture of a device."""

    def __init__(self) -> None:
        """ArchitectureDeterminationError constructor."""
        super().__init__("Could not determine device architecture.")


class ArchitectureNotSupportedError(DeviceApiError):
    """Device architecture is not supported."""

    def __init__(self, arch: str) -> None:
        """ArchitectureNotSupportedError constructor."""
        super().__init__(f"Architecture `{arch}` is not supported.")


class EncodingDecodingError(DeviceApiError):
    """Error during data encoding or decoding."""


class InferenceError(DeviceApiError):
    """Base class for inference exceptions."""


class TFLiteInferenceError(InferenceError):
    """TFlite inference error."""

    def __init__(self, error_message: str) -> None:
        """TFLiteInferenceError constructor."""
        super().__init__(f"TFLite inference failed. Reason: {error_message}")


class LatencyError(DeviceApiError):
    """Base class for latency exceptions."""


class TFLiteLatencyExecutableDownloadError(LatencyError):
    """Executable could not be downloaded."""

    def __init__(self, url: str) -> None:
        """TFLiteLatencyExecutableDownloadError constructor."""
        super().__init__(f"TFLite executable could not be downloded from {url}.")


class TFLiteLatencyError(LatencyError):
    """Error raised during latency assessment."""

    def __init__(self, return_code: int, stderr: str) -> None:
        """TFLiteLatencyError constructor."""
        super().__init__(f"TFLite latency executable failed. Error code {return_code}; {stderr}")


class FingerprintError(DeviceApiError):
    """Error raised during fingerprint retrieval."""
