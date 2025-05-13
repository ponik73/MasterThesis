from typing import Any

from pydantic import BaseModel, Field


class LatencyOutput(BaseModel):
    """Pydantic schema for generic profiling parser output."""

    executable_output: str = Field(description="Output of the framework specific latency evaluator.")


class Batch(BaseModel):
    """Schema for a model input batch."""

    items: list[dict[int, Any]]

    class Config:
        """Pydantic settings."""

        arbitrary_types_allowed = True
