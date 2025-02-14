import shutil
from pathlib import Path
from fastapi import HTTPException, UploadFile

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

    modelDir.mkdir(exist_ok=True, parents=True)

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
    