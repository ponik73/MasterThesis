from pydantic import BaseModel, Field
from typing import Any, Dict, List

class LatencyOutput(BaseModel):
    """Pydantic schema for generic profiling parser output."""

    executableOutput: str = Field(description="Output of the framework specific latency evaluator.")

class Batch(BaseModel):
    items : List[Dict[int, Any]]

    class Config:
        arbitrary_types_allowed = True
    