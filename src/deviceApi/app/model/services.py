from pathlib import Path
from typing import Any

from evaluate.tflite import services as tflite_services
from exceptions import ModelNotFoundError, ModelUploadError
from fastapi import UploadFile


def save_model(custom_name: str, model_upload_file: UploadFile, model_dir: Path) -> None:
    """Save model file.

    :param custom_name: Name under which model will be saved.
    :param model_upload_file: Instance of uploaded model.
    :param model_dir: Path to a model storage directory.
    """
    if model_upload_file.filename is None:
        error_message = "Field 'filename' is not present in uploaded form-data."
        raise ModelUploadError(error_message)

    model_framework = Path(model_upload_file.filename).suffix
    model_path = model_dir / Path(f"{custom_name}{model_framework}")

    model_dir.mkdir(exist_ok=True, parents=True)

    try:
        with model_path.open("wb+") as f:
            f.write(model_upload_file.file.read())
    except OSError as e:
        error_message = "Unable to save model file."
        raise ModelUploadError(error_message) from e


def get_model_input_info_tflite(model_name: str, model_dir: Path) -> list[dict[str, Any]]:
    """Get input nodes info from given tflite model.

    :param model_name: Name under which model is saved.
    :param model_dir: Path to a model storage directory.
    :return: List of dictionaries containing node details
    """
    # Check if model exists:
    model_path = model_dir / f"{model_name}.tflite"
    if not model_path.exists():
        raise ModelNotFoundError(name=model_name, framework="tflite")

    return tflite_services.get_input_info(model_path)
