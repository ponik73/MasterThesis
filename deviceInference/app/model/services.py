import shutil
from pathlib import Path
from fastapi import HTTPException, UploadFile, Depends
from .config import Settings, get_settings
from typing import Annotated

async def saveModel(
        modelUploadFile: UploadFile,
        modelDir: Path
    ):
    """Save model file.

    :param modelUploadFile: Instance of uploaded model.
    :param modelDir: Path to a model storage directory.
    """
    
    if modelUploadFile.filename is None:
        raise HTTPException(status_code=400, detail="Field 'filename' is not present in uploaded form-data.")

    modelPath = modelDir / modelUploadFile.filename

    if checkModelStorage(modelDir):
        await emptyModelStorage(modelDir)

    try:
        with modelPath.open("wb+") as f:
            f.write(modelUploadFile.file.read())

    except OSError as e:
        emptyModelStorage()
        raise HTTPException(status_code=500, detail=f"Unable to save model file. Reason: '{type(e).__name__}'.") from e

    #TODO: Start inference engine here??


async def emptyModelStorage(modelDir: Path):
    """Delete model stored in model storage (only one model is used at a time).
    :param modelDir: Path to a model storage directory.
    """
    
    try:
        shutil.rmtree(modelDir)
        modelDir.mkdir(exist_ok=True, parents=True)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Unable to remove model file. Reason: '{type(e).__name__}'.") from e
    
def checkModelStorage(modelDir: Path) -> Path | None:
    """Checks if a model is stored in storage directory

    Returns:
        Path | None: Path to a model or None if model is not present. 
    """
    if not modelDir.exists():
        modelDir.mkdir(exist_ok=True, parents=True)
        return None
    
    modelPath = None
    for x in modelDir.iterdir():
        modelPath = x
        break

    return modelPath