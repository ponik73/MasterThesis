from pathlib import Path
import subprocess
from fastapi import HTTPException, UploadFile
from model import config as modelConfig 

async def latencyAssessmentTFlite(
        modelPath: Path,
        latencyExecutablePath: Path
    ):

    if not latencyExecutablePath.exists():
        raise HTTPException(status_code=500, detail=f"Latency executable not found.")
    
    try:

        ############ This is tflite specific (benchmark_model):
        cmd = f'{latencyExecutablePath.as_posix()} --graph={modelPath.as_posix()}'
        # assessmentProcess = subprocess.run(f'{latencyExecutablePath.as_posix()} --graph={modelPath.as_posix()}', capture_output=True, text=True, shell = True)
        assessmentProcess = subprocess.run('ver', capture_output=True, text=True, shell = True)
        #####################################
        

        if assessmentProcess.returncode != 0 or not assessmentProcess.stdout:
            raise Exception("TODO exceptions.py file (or maybe here) - LatencyAssessmentError")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during latency assessment Reason: '{type(e).__name__}'.") from e

    return assessmentProcess.stdout