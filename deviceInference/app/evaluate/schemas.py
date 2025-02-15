from pydantic import BaseModel, Field, HttpUrl
from typing import Literal

class LatencyRequestBody(BaseModel):
    """Pydantic schema for GET /evaluate endpoint request body."""

    latencyExecutable: str = Field(description="Path to a framework specific latency evaluator executable.")

class LatencyOutput(BaseModel):
    """Pydantic schema for generic profiling parser output."""

    executableOutput: str = Field(description="Output of the framework specific latency evaluator.")