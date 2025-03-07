from pathlib import Path
from typing import Dict, Literal
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings

class ArchitectureExecutableMapping(BaseSettings):
    """Mapping of linux architectures to the source of the TFLite benchmark_model executables."""
    mappings: Dict[str, str] = Field(
        default_factory=lambda: {
            "aarch64": "https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model"
        }
    )
    EXECUTABLE_NAME: Literal["benchmark_model"] = "benchmark_model"

def getExecutableUrl(architecture: str) -> str | None:
    return ArchitectureExecutableMapping().mappings.get(architecture, None)

def getExecutablePath(executableDir: Path) -> Path | None:
    # path = executableDir / ArchitectureExecutableMapping().EXECUTABLE_NAME
    # if path.exists() and path.is_file():
    #     return path
    # return None
    return executableDir / ArchitectureExecutableMapping().EXECUTABLE_NAME