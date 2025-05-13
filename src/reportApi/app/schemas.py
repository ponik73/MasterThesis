from database.models import DatasetBase, DeviceBase, MetricBase, ModelBase
from pydantic import BaseModel


class GenericAPIResponse(BaseModel):
    """Generic model for the API response."""


class SuccessAPIResponse(GenericAPIResponse):
    """API reponse model for success message."""

    message: str


class ErrorAPIResponse(GenericAPIResponse):
    """Generic API response model for error message."""

    error_message: str


class RegisterRunInput(BaseModel):
    """Class representing `/run` endpoint payload."""

    device: DeviceBase
    model: ModelBase
    latency_metrics: list[MetricBase]
    dataset: DatasetBase
    accuracy_metrics: list[MetricBase]


class RegisterRunOutput(BaseModel):
    """Class representing `/run` endpoint output."""

    run_id: int


class GetReportInput(BaseModel):
    """Class representing `/report` endpoint query."""

    run_ids: list[int]
