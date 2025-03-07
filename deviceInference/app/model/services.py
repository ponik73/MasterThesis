from pathlib import Path
from fastapi import HTTPException, UploadFile

from evaluate.tflite import services as tfliteServices

def saveModel(
        customName: str,
        modelUploadFile: UploadFile,
        modelDir: Path
    ):
    """Save model file.

    :param customName: Name under which model will be saved.
    :param modelUploadFile: Instance of uploaded model.
    :param modelDir: Path to a model storage directory.
    """
    
    if modelUploadFile.filename is None:
        raise HTTPException(status_code=400, detail="Field 'filename' is not present in uploaded form-data.")
    
    if not customName:
        raise HTTPException(status_code=400, detail="Custom name is not specified.")

    modelFramework = Path(modelUploadFile.filename).suffix
    modelPath = modelDir / Path(f'{customName}{modelFramework}')
    
    modelDir.mkdir(exist_ok=True, parents=True)

    try:
        with modelPath.open("wb+") as f:
            f.write(modelUploadFile.file.read())
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Unable to save model file. Reason: '{type(e).__name__}'.") from e
    
def getModelInputInfoTflite(modelName: str, modelDir: Path):
    # Check if model exists:
    modelPath = modelDir / f"{modelName}.tflite"
    if not modelPath.exists():
        raise HTTPException(status_code=500, detail=f"Model `{modelName}.tflite` not found.")
    
    return tfliteServices.getInputInfo(modelPath)