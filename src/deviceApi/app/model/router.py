from typing import Annotated

from config import Settings, get_settings
from fastapi import APIRouter, Depends, File, Form, UploadFile

from .schemas import InputInfoTfliteOutput
from .services import get_model_input_info_tflite, save_model

model_router = APIRouter(
    prefix="/model",
    tags=["model"],
)


@model_router.post("/upload", description="Upload model.")
async def upload_model(
    custom_name: Annotated[str, Form()],
    model_file: Annotated[UploadFile, File(description="Uploaded model file.")],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Store given model on device.

    :param custom_name: Name model will be saved as
    :param settings: settings dependency injection
    :param model_file: Reference to uploaded model file
    """
    return save_model(custom_name, model_file, settings.MODEL_DIR)


@model_router.get(
    "/input-info/tflite", description="Get info about input of the TFLite model.", response_model=InputInfoTfliteOutput
)
async def get_input_info_tflite(
    model_custom_name: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> InputInfoTfliteOutput:
    """Get information about tflite model's input nodes.

    :param model_custom_name: Model name
    :param settings: settings dependency injection
    """
    return InputInfoTfliteOutput.from_list(data_list=get_model_input_info_tflite(model_custom_name, settings.MODEL_DIR))
