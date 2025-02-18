from typing import Annotated
from fastapi import APIRouter, Depends, File, UploadFile, Form

from config import Settings, getSettings
from .services import saveModel

modelRouter = APIRouter(
    prefix="/model",
    tags=["model"],
)

@modelRouter.post(
        "/",
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
