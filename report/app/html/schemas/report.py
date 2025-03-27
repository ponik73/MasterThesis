from database.models import Dataset, Device, Metric, Model
from pydantic import BaseModel


class RunResults(BaseModel):
    """Report template object used for display of a run."""

    model: Model
    latency_results: list[Metric]
    dataset: Dataset
    accuracy_results: list[Metric]


class DeviceResults(BaseModel):
    """Report template object used for display of a device."""

    device: Device
    results: list[RunResults]
