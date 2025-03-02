from typing import Annotated
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException

from config import Settings, getSettings
from .services import saveModel, getModelInputInfoTflite
from .schemas import InputInfoTfliteOutput

modelRouter = APIRouter(
    prefix="/model",
    tags=["model"],
)

@modelRouter.post(
        "/upload",
        description="Upload model."
)
async def uploadModel(
    customName: Annotated[str, Form()],
    modelFile: Annotated[UploadFile, File(description="Uploaded model file.")],
    settings: Annotated[Settings, Depends(getSettings)],
):
    """Upload model

    Args:
        customName (Annotated[str, Form): Name model will be saved as.
        modelFile (Annotated[UploadFile, File, optional): Reference to uploaded model file.
    """
    return saveModel(customName, modelFile, settings.MODEL_DIR)

@modelRouter.get(
        "/input-info/tflite",
        description="Get info about input of the TFLite model.",
        response_model=InputInfoTfliteOutput
)
async def getInputInfoTflite(
    modelCustomName: str,
    settings: Annotated[Settings, Depends(getSettings)],
):
    return InputInfoTfliteOutput.from_list(
        data_list=getModelInputInfoTflite(modelCustomName, settings.MODEL_DIR)
    )