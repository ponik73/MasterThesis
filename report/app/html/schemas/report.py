from typing import List
from pydantic import BaseModel

from database.models import Model, Metric, Dataset, Device


class RunResults(BaseModel):
        model: Model
        latency_results: List[Metric]
        dataset: Dataset
        accuracy_results: List[Metric]

class DeviceResults(BaseModel):
    device: Device
    results: List[RunResults]