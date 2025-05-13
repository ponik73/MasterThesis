import subprocess
from pathlib import Path
from typing import Any

from exceptions import (
    ArchitectureDeterminationError,
    ArchitectureNotSupportedError,
    TFLiteLatencyError,
    TFLiteLatencyExecutableDownloadError,
)

from ..schemas import Batch
from .config import get_executable_path, get_executable_url
from .interpreter import tflite_interpreter


def latency_assessment_tflite(model_path: Path, executable_dir: Path) -> str:
    """Latency assessment of specified model using `benchmark_model` executable.

    :param model_path: path of evaluated model
    :param executable_dir: path of directory where executable is (will be) stored
    :return: Raw output of the `benchmark_model` executable.
    """
    # Check if tflite latency executable is present:
    executable_dir.mkdir(parents=True, exist_ok=True)
    executable_path = get_executable_path(executable_dir)

    if not executable_path.exists():  # Download the executable:
        architecture = _determine_architecture()
        # Source of the executable:
        url = get_executable_url(architecture)
        if not url:
            raise ArchitectureNotSupportedError(architecture)
        # Download:
        _fetch_latency_executable(url, executable_path)
        if not executable_path.exists():
            raise TFLiteLatencyExecutableDownloadError(url=url)

    # Run the assessment:
    assessment_process = subprocess.run(
        f"{executable_path.as_posix()} --graph={model_path.as_posix()}",
        capture_output=True,
        text=True,
        shell=True,
        check=False,
    )
    if assessment_process.returncode != 0 or not assessment_process.stdout:
        raise TFLiteLatencyError(return_code=assessment_process.returncode, stderr=assessment_process.stderr)

    return assessment_process.stdout


def accuracy_assessment_tflite(model_path: Path, batch: Batch) -> list[dict[int, Any]]:
    """Execute inference of a batch over specified model.

    :param model_path: model that will be infered
    :param batch: mode input batch
    :return: List of dictionaries that contain output node value mapped to an index.
    """
    # Check if interpeter instance is running, if not create an instance:
    tflite_interpreter.start(model_path)

    return tflite_interpreter.inference(batch)


def get_input_info(model_path: Path) -> list[dict[str, Any]]:
    """Get details of specified model's input nodes.

    :param model_path: model path
    :return: List of dictionaries containing node details
    """
    if tflite_interpreter.model_path == model_path:
        return tflite_interpreter.get_input_details()

    tflite_interpreter.start(model_path)
    return tflite_interpreter.get_input_details()


def _determine_architecture() -> str:
    """Get name of the device's architecture.

    :return: architecture literal
    """
    cmd_result = subprocess.run("uname -m", capture_output=True, text=True, shell=True, check=False)
    if cmd_result.returncode != 0 or cmd_result.stderr:
        raise ArchitectureDeterminationError
    return cmd_result.stdout.strip()


def _fetch_latency_executable(url: str, executable_path: Path) -> None:
    """Download and store `benchmark_model` executable.

    :param url: executable url
    :param executable_path: path of file where executable will be stored
    """
    import urllib.request

    if not url.startswith(("http:", "https:")):
        error_message = "URL must start with 'http:' or 'https:'"
        raise ValueError(error_message)
    urllib.request.urlretrieve(url, executable_path)
    executable_path.chmod(0o755)
