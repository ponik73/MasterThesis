from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Depends, File, UploadFile, Form
from .schemas import LatencyRequestBody, LatencyOutput
from .services import latencyAssessmentTFlite, isModelPresent
from model.config import Settings, get_settings
# from evaluate.schemas import AhojRequestBody, AhojOutput#, AhojOutput

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
    latencyExecutable: Annotated[UploadFile, File(description="Executable that evaluates latency (e.g. benchmark_model for TFLite).")],
    settings: Annotated[Settings, Depends(get_settings)]
):    
    # # Check if the model is present in the storage:
    # modelPath = checkModelStorage(settings.MODEL_DIR)
    # if not modelPath or not modelPath.exists() or not modelPath.is_file():
    #     raise Exception("TODO exceptions.py file (or maybe here) - NoModelFoundError")
    
    # # Check if the model's framework is tflite:
    # if modelPath.suffix != ".tflite":
    #     raise Exception("TODO exceptions.py file (or maybe here) - MLFramworkNotSupportedError")
    
    modelPath = isModelPresent(settings.MODEL_DIR, ".tflite")

    serviceFuncArgs = {
        "modelPath": modelPath,
        "bechmarkmodelUpload": latencyExecutable,
        "tempDirPath": settings.TEMP_DIR
    }

    return LatencyOutput(executableOutput=await latencyAssessmentTFlite(**serviceFuncArgs))

@evaluateRouter.post(
    "/accuracy/tflite",
    description="",
    # response_model=AhojOutput
)
async def evaluateAccuracyTFLite(
    settings: Annotated[Settings, Depends(get_settings)]
):
    modelPath = isModelPresent(settings.MODEL_DIR, ".tflite")
    
    return {"ahoj": modelPath.as_posix()}