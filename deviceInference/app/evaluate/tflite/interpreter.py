from pathlib import Path
import numpy as np
# import tflite_runtime.interpreter as tflite
from tensorflow import lite as tflite
from fastapi import HTTPException
from typing import List, Any, Dict
from ..schemas import Batch

class TfliteInterpreter():
    def __init__(self):
        self.modelPath: Path | None = None
        self.interpreter : tflite.Interpreter | None = None
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
            self.detailsInput : List[Dict[str, Any]] = self.interpreter.get_input_details()
            self.detailsOutput = self.interpreter.get_output_details()
            self.modelPath = modelPath
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unable to initiate TFLite interpreter. Reason: '{type(e).__name__}'.") from e
    
    def getInputDetails(self):        
        if not self.detailsInput:
            raise HTTPException(status_code=500, detail=f"Unable to execyte TFLite inference.")
        
        return self.detailsInput
    
    def getOutputIndexes(self) -> List[int]:
        if not self.detailsOutput:
            raise HTTPException(status_code=500, detail=f"Unable to execyte TFLite inference.")
        
        return [x['index'] for x in self.detailsOutput]
    
    def inference(self, batch: Batch):
        try:
            self.interpreter.allocate_tensors()

            results : List[Dict[int, np.typing.NDArray[Any]]] = [] # Batch results
            # Execute the inference of batch:
            for item in batch.items:
                # Fed data to each input:
                for inputIdx, data in item.items():
                    self.interpreter.set_tensor(inputIdx, np.array([data]))
                # Inference:
                self.interpreter.invoke()
                # Retrieve inference result:
                outputIndexes = self.getOutputIndexes()
                res = {} # Results of one item in the batch
                for outputIdx in outputIndexes:
                    res[outputIdx] = self.interpreter.get_tensor(outputIdx)
                # Add the results of item to the batch result list:
                results.append(res)

        except Exception as e:
            raise Exception("TODO exceptions.py file (or maybe here) - TFLiteInferenceError" + e)

        return results
    
tfliteInterpreter = TfliteInterpreter()