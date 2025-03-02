from pydantic import BaseModel, Field
import numpy as np
from typing import Any, Dict, List

class LatencyOutput(BaseModel):
    """Pydantic schema for generic profiling parser output."""

    executableOutput: str = Field(description="Output of the framework specific latency evaluator.")

class Batch(BaseModel):
    items : List[Dict[int, np.typing.NDArray[Any]]]

    class Config:
        arbitrary_types_allowed = True
    