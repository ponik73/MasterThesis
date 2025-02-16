from typing import Annotated
from fastapi import APIRouter, Depends, File, UploadFile
from starlette import status

from .config import Settings, get_settings
from .services import saveModel, emptyModelStorage

modelRouter = APIRouter(
    prefix="/model",
    tags=["model"],
)

@modelRouter.post(
        "/",
        description="Upload model and save metadata to database."
)
async def upload_model(
    modelFile: Annotated[UploadFile, File(description="Uploaded model file.")],
    settings: Annotated[Settings, Depends(get_settings)]
):
    """Upload model and save metadata to database.

    :param model_file: Reference to uploaded model file.
    """
    return await saveModel(modelFile, settings.MODEL_DIR)


@modelRouter.delete(
    "/",
    description="Delete model and all associated metadata.",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_model(
    settings: Annotated[Settings, Depends(get_settings)]
):
    """Delete stored model."""
    return await emptyModelStorage(settings.MODEL_DIR)
