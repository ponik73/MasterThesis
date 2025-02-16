from pathlib import Path
import shutil
import subprocess
from fastapi import HTTPException, UploadFile
from model import config as modelConfig 

async def latencyAssessmentTFlite(
        modelPath: Path,
        bechmarkmodelUpload: UploadFile,
        tempDirPath: Path
    ):

    if bechmarkmodelUpload.filename is None:
        raise HTTPException(status_code=400, detail="Field 'filename' is not present in uploaded form-data.")

    # Save executable:
    tempDirPath.mkdir(exist_ok=True, parents=True)
    latencyExecutablePath = tempDirPath / bechmarkmodelUpload.filename
    try:
        with latencyExecutablePath.open("wb+") as f:
            f.write(bechmarkmodelUpload.file.read())
        latencyExecutablePath.chmod(0o755)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Unable to save executable. Reason: '{type(e).__name__}'.") from e

    # Run the assessment:
    try:
        assessmentProcess = subprocess.run(f'{latencyExecutablePath.as_posix()} --graph={modelPath.as_posix()}', capture_output=True, text=True, shell = True)
        if assessmentProcess.returncode != 0 or not assessmentProcess.stdout:
            raise Exception(str(assessmentProcess.returncode) + assessmentProcess.stderr + "\n" + assessmentProcess.stdout)
            # raise Exception("TODO exceptions.py file (or maybe here) - LatencyAssessmentError")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during latency assessment Reason: '{e}'.") from e

    # TODO: remove the executable

    return assessmentProcess.stdout
