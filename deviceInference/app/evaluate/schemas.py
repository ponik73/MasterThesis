from pydantic import BaseModel, Field

class LatencyOutput(BaseModel):
    """Pydantic schema for generic profiling parser output."""

    executableOutput: str = Field(description="Output of the framework specific latency evaluator.")