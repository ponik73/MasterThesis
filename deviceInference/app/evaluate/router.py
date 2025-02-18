from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from .schemas import LatencyOutput
from .services import latencyAssessment, accuracyAssessment
from config import Settings, getSettings

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
    latencyExecutable: Annotated[UploadFile, File(description="Executable that evaluates latency (e.g. benchmark_model for TFLite).")],
    settings: Annotated[Settings, Depends(getSettings)],
):
    # TODO: Based on Linux (aarch64, arm, x86-64) get the benchmark_model executable (Docker process).

    modelPath = settings.MODEL_DIR / f'{modelCustomName}.tflite'
    if not modelPath or not modelPath.exists() or not modelPath.is_file():
        raise Exception("TODO exceptions.py file (or maybe here) - ModelNotFoundError")
    
    serviceFuncArgs = {
        "frameworkNN": ".tflite",
        "modelPath": modelPath,
        "bechmarkmodelUpload": latencyExecutable,
        "tempDirPath": settings.TEMP_DIR
    }

    return LatencyOutput(executableOutput=latencyAssessment(**serviceFuncArgs))

@evaluateRouter.post(
    "/accuracy/tflite"#,
    # description="",
    # response_model=AhojOutput # TODO: output standard
)
async def evaluateAccuracyTFLite(
    # modelCustomName: Annotated[str, Form()],
    modelCustomName: str,
    batch: Annotated[bytes, File()],
    settings: Annotated[Settings, Depends(getSettings)],
):
    # Check if model is present for inference:
    modelPath = settings.MODEL_DIR / f'{modelCustomName}.tflite'
    if not modelPath or not modelPath.exists() or not modelPath.is_file():
        raise HTTPException(status_code=500, detail=f"Model not found.")

    result = accuracyAssessment(".tflite", modelPath, batch)

    return {"aa": str(result)}
