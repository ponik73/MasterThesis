from pathlib import Path


class ModelEvaluatorError(Exception):
    """Base class for Model Evaluator module errors."""


class DeviceApiError(ModelEvaluatorError):
    """Base class for errors occuring during device API calls."""


class DeviceApiInitError(DeviceApiError):
    """Could not init device API."""


class ModelUploadError(DeviceApiError):
    """Model upload call error."""

    def __init__(self, model_local_path: Path, api_url: str) -> None:
        """ModelUploadError constructor."""
        super().__init__(f"Could not upload {model_local_path} to {api_url}")


class OndeviceInferenceError(DeviceApiError):
    """Inference results were not recieved."""

    def __init__(self, model_local_path: Path, api_url: str) -> None:
        """OndeviceInferenceError constructor."""
        super().__init__(f"Model `{model_local_path}` at {api_url} failed.")


class FingerprintError(DeviceApiError):
    """Fingerprint call error."""

    def __init__(self, url: str) -> None:
        """FingerprintError constructor."""
        super().__init__(f"Unable to retrieve fingerprint of device at `{url}`.")


class ModelInfoError(DeviceApiError):
    """Model info call error."""


class LatencyAssessmentError(DeviceApiError):
    """Latency assessment call error."""


class AccuracyAssessmentError(DeviceApiError):
    """Accuracy assessment call error."""


class FrameworkNotSupportedError(ModelEvaluatorError):
    """Neural network framework is not supported."""

    def __init__(self, framework: str) -> None:
        """FrameworkNotSupportedError constructor."""
        super().__init__(f"Neural network framework `{framework}` is not supported.")


class ReportApiError(ModelEvaluatorError):
    """Base class for errors occuring during report API calls."""


class RegisterRunCallError(ReportApiError):
    """Register run call error."""


class GenerateReportCallError(ReportApiError):
    """Register run call error."""
