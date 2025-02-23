from pathlib import Path
import numpy as np
import pickle
import codecs
from typing import Literal
from fastapi import HTTPException, UploadFile
from .tflite.services import latencyAssessmentTFlite, accuracyAssessmentTFlite

def latencyAssessment(
        frameworkNN: Literal[".tflite"],
        modelPath: Path,
        bechmarkmodelUpload: UploadFile,
        tempDirPath: Path
):
    latencyAssessmentFunctions = {
        ".tflite": latencyAssessmentTFlite
    }

    if frameworkNN not in latencyAssessmentFunctions.keys():
        raise HTTPException(status_code=500, detail=f"Framework `{frameworkNN}` not supported.")
    
    return latencyAssessmentFunctions[frameworkNN](modelPath, bechmarkmodelUpload, tempDirPath)

def accuracyAssessment(
        frameworkNN: Literal[".tflite"],
        modelPath: Path,
        encodedBatch: bytes
):
    assessmentFunctions = {
        ".tflite": accuracyAssessmentTFlite
    }

    # Decode batch data:
    decodedBatch = decodeBatch(encodedBatch)

    if frameworkNN not in assessmentFunctions.keys():
        raise HTTPException(status_code=500, detail=f"Framework `{frameworkNN}` not supported.")
    
    # Execute inference:
    inferenceResult = assessmentFunctions[frameworkNN](modelPath, decodedBatch)
    
    return encodeModelOutputs(inferenceResult)

def decodeBatch(encodedBatch: bytes) -> np.array:
    try:
        return np.array(pickle.loads(codecs.decode(encodedBatch, "base64")))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during decoding batch data. Reason: '{e}'.") from e

def encodeModelOutputs(modelOutputs) -> bytes:
    return codecs.encode(pickle.dumps(modelOutputs, protocol=pickle.HIGHEST_PROTOCOL), "base64")