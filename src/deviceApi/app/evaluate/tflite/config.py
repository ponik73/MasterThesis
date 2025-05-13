from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class ArchitectureExecutableMapping(BaseSettings):
    """Mapping of linux architectures to the source of the TFLite benchmark_model executables."""

    mappings: dict[str, str] = Field(
        default_factory=lambda: {
            "aarch64": "https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model"
        }
    )
    EXECUTABLE_NAME: Literal["benchmark_model"] = "benchmark_model"


def get_executable_url(architecture: str) -> str | None:
    """Return url of a remote executable file based on model architecture.

    :param architecture: architecture identifier
    :return: Url of a remote executable file
    """
    return ArchitectureExecutableMapping().mappings.get(architecture, None)


def get_executable_path(executable_dir: Path) -> Path | None:
    """Return path of a latency executable.

    :param executable_dir: path of directory where executable is expected to be stored
    :return: Executable path.
    """
    # path = executable_dir / ArchitectureExecutableMapping().EXECUTABLE_NAME
    # if path.exists() and path.is_file():
    #     return path
    # return None
    return executable_dir / ArchitectureExecutableMapping().EXECUTABLE_NAME
