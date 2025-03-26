from typing import Annotated
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from fastapi.responses import Response

from .schemas import LatencyOutput
from .services import latencyAssessment, accuracyAssessment
from config import Settings, getSettings
from exceptions import ModelNotFoundError

evaluateRouter = APIRouter(
    prefix="/evaluate",
    tags=["evaluate"]
)

@evaluateRouter.post(
    "/latency/tflite",
    description="Evaluate latency of uploaded TFLite model.",
    response_model=LatencyOutput
)
async def evaluateLatencyTFLite(
    modelCustomName: Annotated[str, Form()],
    settings: Annotated[Settings, Depends(getSettings)],
):
    modelPath = settings.MODEL_DIR / f'{modelCustomName}.tflite'
    if not modelPath or not modelPath.exists() or not modelPath.is_file():
        raise ModelNotFoundError(name=modelCustomName, framework='tflite')
    
    serviceFuncArgs = {
        "frameworkNN": ".tflite",
        "modelPath": modelPath,
        "tempDirPath": settings.TEMP_DIR
    }

    return LatencyOutput(executableOutput=latencyAssessment(**serviceFuncArgs))

@evaluateRouter.post(
    "/accuracy/tflite",
    description=""
)
async def evaluateAccuracyTFLite(
    batch: Annotated[bytes, File()],
    modelCustomName: Annotated[str, Form()],
    settings: Annotated[Settings, Depends(getSettings)],
):
    # Check if model is present for inference:
    modelPath = settings.MODEL_DIR / f'{modelCustomName}.tflite'
    if not modelPath or not modelPath.exists() or not modelPath.is_file():
        raise HTTPException(status_code=500, detail=f"Model not found.")

    return Response(
        content=accuracyAssessment(".tflite", modelPath, batch),
        media_type="application/octet-stream"
    )
