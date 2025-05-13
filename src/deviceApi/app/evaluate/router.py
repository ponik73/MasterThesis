from typing import Annotated

from config import Settings, get_settings
from exceptions import ModelNotFoundError
from fastapi import APIRouter, Depends, File, Form
from fastapi.responses import Response

from .schemas import LatencyOutput
from .services import accuracy_assessment, latency_assessment

evaluate_router = APIRouter(prefix="/evaluate", tags=["evaluate"])


@evaluate_router.post(
    "/latency/tflite", description="Evaluate latency of uploaded TFLite model.", response_model=LatencyOutput
)
async def evaluate_latency_tflite(
    model_custom_name: Annotated[str, Form()],
    settings: Annotated[Settings, Depends(get_settings)],
) -> LatencyOutput:
    """Evaluate latency of uploaded TFLite model.

    :param model_custom_name: custom model name
    :param settings: settings dependency injection
    :return: Profiling executable output.
    """
    model_path = settings.MODEL_DIR / f"{model_custom_name}.tflite"
    if not model_path or not model_path.exists() or not model_path.is_file():
        raise ModelNotFoundError(name=model_custom_name, framework="tflite")

    service_func_args = {"framework_nn": ".tflite", "model_path": model_path, "temp_dir_path": settings.TEMP_DIR}

    return LatencyOutput(executable_output=latency_assessment(**service_func_args))


@evaluate_router.post("/accuracy/tflite", description="")
async def evaluate_accuracy_tflite(
    batch: Annotated[bytes, File()],
    model_custom_name: Annotated[str, Form()],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Execute inference of a batch on given model.

    :param batch: Input batch
    :param model_custom_name: Name of the model
    :param settings: Settings dependency injection
    :return: Model outputs as list of dictionaries with keys corresponding to output indexes and values
      corresponding to a output values.
    """
    # Check if model is present for inference:
    model_path = settings.MODEL_DIR / f"{model_custom_name}.tflite"
    if not model_path or not model_path.exists() or not model_path.is_file():
        raise ModelNotFoundError(name=model_custom_name, framework="tflite")

    return Response(content=accuracy_assessment(".tflite", model_path, batch), media_type="application/octet-stream")
