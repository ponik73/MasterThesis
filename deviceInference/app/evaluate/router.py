from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Depends, File, UploadFile, Form
from .schemas import LatencyRequestBody, LatencyOutput
from .services import latencyAssessmentTFlite
from model.services import checkModelStorage
from model.config import Settings, get_settings
# from evaluate.schemas import AhojRequestBody, AhojOutput#, AhojOutput

evaluateRouter = APIRouter(
    prefix="/evaluate",
    tags=["evaluate"]
)

@evaluateRouter.get(
    "/",
    description="Evaluate latency of uploaded model.",
    response_model=LatencyOutput
)
async def evaluateLatency(
    latencyExecutablePath: str,
    settings: Annotated[Settings, Depends(get_settings)]
):
    modelPath = checkModelStorage(settings.MODEL_DIR)
    if not modelPath or not modelPath.exists() or not modelPath.is_file():
        raise Exception("TODO exceptions.py file (or maybe here) - NoModelFoundError")
    
    modelFramework = modelPath.suffix[1:]
    if modelFramework not in settings.SUPPORTED_FRAMEWORKS:
        raise Exception("TODO exceptions.py file (or maybe here) - MLFramworkNotSupportedError")

    # TODO: Generalization - based on NN framework select framework specific (e.g. latencyAssessmentTFLite) function from services; some mapping
    serviceFunc = latencyAssessmentTFlite
    serviceFuncArgs = {
        "modelPath": modelPath,
        "latencyExecutablePath": Path(latencyExecutablePath)
    }

    return LatencyOutput(executableOutput=await serviceFunc(**serviceFuncArgs))

# @evaluateRouter.post(
#     "/",
#     description="",
#     # response_model=AhojOutput
# )