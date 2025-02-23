from pathlib import Path
import numpy as np
# import tflite_runtime.interpreter as tflite
from tensorflow import lite as tflite
from fastapi import HTTPException
from typing import List

class TfliteInterpreter():
    def __init__(self):
        self.modelPath: Path | None = None
        self.interpreter = None
        self.detailsInput = None
        self.detailsOutput = None

    def start(self, modelPath: Path) -> bool:
        # Return if interpreter is running:
        if self.interpreter and self.modelPath == modelPath:
            return
        
        # Init interpreter:
        try:
            self.interpreter = tflite.Interpreter(
                model_path=modelPath.as_posix())#,
                # experimental_delegates=ext_delegate)
            self.detailsInput = self.interpreter.get_input_details()
            self.detailsOutput = self.interpreter.get_output_details()
            self.modelPath = modelPath
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unable to initiate TFLite interpreter. Reason: '{type(e).__name__}'.") from e
    
    def getInputDetails(self):
        # TODO: multiple inputs
        
        if not self.detailsInput:
            raise HTTPException(status_code=500, detail=f"Unable to execyte TFLite inference.")
        
        return self.detailsInput[0]['index']
    
    def getOutputIndexes(self) -> List[int]:
        if not self.detailsOutput:
            raise HTTPException(status_code=500, detail=f"Unable to execyte TFLite inference.")
        
        return [x['index'] for x in self.detailsOutput]
    
    def inference(self, batch: np.array):
        # TODO: multiple inputs

        try:
            # Identify input:
            inputIdx = self.getInputDetails()

            # Resize input for the batch:
            self.interpreter.resize_tensor_input(inputIdx, batch.shape)
            self.interpreter.allocate_tensors()

            # Execute the inference of batch:
            self.interpreter.set_tensor(inputIdx, batch)
            self.interpreter.invoke()

            # Retrieve inference result:
            results = {}
            outputIndexes = self.getOutputIndexes()
            for outputIdx in outputIndexes:
                results[outputIdx] = self.interpreter.get_tensor(outputIdx)

        except Exception as e:
            raise Exception("TODO exceptions.py file (or maybe here) - TFLiteInferenceError" + e)

        return results
    
tfliteInterpreter = TfliteInterpreter()