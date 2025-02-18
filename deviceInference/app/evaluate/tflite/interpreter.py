from pathlib import Path
import numpy as np
# import tflite_runtime.interpreter as tflite
from tensorflow import lite as tflite
from fastapi import HTTPException

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
            raise HTTPException(status_code=500, detail=f"Unable to TFLite interpreter not initiated.")
        
        return {
            "index": self.detailsInput[0]['index'],
            "shape": np.array(self.detailsInput[0]['shape']),
            "dtype": np.dtype(self.detailsInput[0]['dtype']).name 
        }
    
    def inference(self, inputIdx: int, decodedBatch: np.array):
        # TODO: multiple inputs

        try:
            # Resize input for the batch:
            self.interpreter.resize_tensor_input(inputIdx, decodedBatch.shape)
            self.interpreter.allocate_tensors()

            # Execute the inference of batch:
            # decodedBatch = np.ones([1,224,224,3], dtype='float32')
            self.interpreter.set_tensor(inputIdx, decodedBatch)
            self.interpreter.invoke()

            # Retrieve inference result:
            result = self.interpreter.get_tensor(self.detailsOutput[0]['index'])
        except Exception as e:
            raise Exception("TODO exceptions.py file (or maybe here) - TFLiteInferenceError" + e)

        return result
    
tfliteInterpreter = TfliteInterpreter()