from pydantic import BaseModel, Field

from typing import List
from database.models import DeviceBase, ModelBase, DatasetBase, MetricBase

class GenericAPIResponse(BaseModel):
    """Generic model for the API response."""


class SuccessAPIResponse(GenericAPIResponse):
    """API reponse model for success message."""

    message: str


class ErrorAPIResponse(GenericAPIResponse):
    """Generic API response model for error message."""

    error_message: str

class RegisterRunInput(BaseModel):
    device: DeviceBase
    model: ModelBase
    latency_metrics: List[MetricBase]
    dataset: DatasetBase
    accuracy_metrics: List[MetricBase]

class RegisterRunOutput(BaseModel):

    run_id: int

class GetReportInput(BaseModel):
    run_ids: List[int]
