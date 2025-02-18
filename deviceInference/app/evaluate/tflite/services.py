from pathlib import Path
from fastapi import HTTPException, UploadFile
import subprocess
import numpy as np

from .interpreter import tfliteInterpreter
from .config import getExecutablePath, getExecutableUrl

def latencyAssessmentTFlite(
        modelPath: Path,
        executableDir: Path
    ):
    # Check if tflite latency executable is present:
    executableDir.mkdir(parents=True, exist_ok=True)
    executablePath = getExecutablePath(executableDir)

    if not executablePath.exists(): # Download the executable:
        architecture = _determineArchitecture()
        # Source of the executable:
        url = getExecutableUrl(architecture)
        if not url:
            raise HTTPException(status_code=500, detail=f"Architecture `{architecture}` not supported.")
        # Download:
        _fetchLatencyExecutable(url, executablePath)
        if not executablePath.exists():
            raise HTTPException(status_code=500, detail=f"Unable to download executable.")

    # Run the assessment:
    try:
        assessmentProcess = subprocess.run(f'{executablePath.as_posix()} --graph={modelPath.as_posix()}', capture_output=True, text=True, shell = True)
        if assessmentProcess.returncode != 0 or not assessmentProcess.stdout:
            raise Exception(str(assessmentProcess.returncode) + assessmentProcess.stderr + "\n" + assessmentProcess.stdout)
            # raise Exception("TODO exceptions.py file (or maybe here) - LatencyAssessmentError")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during latency assessment Reason: '{e}'.") from e

    return assessmentProcess.stdout

def accuracyAssessmentTFlite(modelPath: Path, batch: np.array):
    # Check if interpeter instance is running, if not create an instance:
    tfliteInterpreter.start(modelPath)

    # So far only one input models. TODO: multiple inputs; TODO: move into services.py
    singleInput = tfliteInterpreter.getInputDetails()

    return tfliteInterpreter.inference(singleInput["index"], batch)

# TODO: maybe move somewhere more general
def _determineArchitecture() -> str:
    cmdResult = subprocess.run('uname -m', capture_output=True, text=True, shell = True)
    if cmdResult.returncode != 0 or cmdResult.stderr:
        raise HTTPException(status_code=500, detail=f"Could not determine device architecture.")
    return cmdResult.stdout.strip()

def _fetchLatencyExecutable(url: str, executablePath: Path):
    import urllib.request
    try:
        urllib.request.urlretrieve(url, executablePath)
        executablePath.chmod(0o755)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to download executable. Reason: '{type(e).__name__}'.") from e