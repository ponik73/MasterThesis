import tflite_runtime.interpreter as tflite
# from tensorflow import lite as tflite

##################################################
import sys
# REMOVE:
class State:
    def __init__(self):
        self.interpreter : tflite.Interpeter | None = None
        self.detailsInput = None
        self.detailsOutput = None

class App:
    def __init__(self):
        self.state = State()
##################################################

app = App()

# TODO: POST request
def uploadModel():
    # Recieve model:
    modelPath = sys.argv[1]

    # Initialize interpreter for the model:
    # TODO: Specify in config:
    ext_delegate = tflite.load_delegate('/usr/lib/libvx_delegate.so') #None
    num_threads = None

    app.state.interpreter = tflite.Interpreter(
        model_path=modelPath,
        experimental_delegates=ext_delegate,
        num_threads=num_threads)
    app.state.interpreter.allocate_tensors()

    app.state.detailsInput = app.state.interpreter.get_input_details()
    app.state.detailsOutput = app.state.interpreter.get_output_details()

    result = {
        "modelRemotePath": modelPath,
        "detailsInput": app.state.detailsInput,
        "detailsOutput": app.state.detailsOutput
    }
    print(result) # Return to the API caller

# TODO: GET/PUT request
def removeModel():
    app.state.interpreter = None
    app.state.detailsInput = None
    app.state.detailsOutput = None

# TODO: POST request
def infer(inputs):
    # Feed the model inputs:
    for inputIdx in inputs.keys():
        app.state.interpreter.set_tensor(int(inputIdx), inputs[inputIdx])
    # Inference:
    app.state.interpreter.invoke()
    # Get the output:
    outputIdxs = [x['index'] for x in app.state.detailsOutput] # TODO: Maybe replace app.state.detailsOutput with app.state.outputIdxs
    
    outputs = {}
    for outputIdx in outputIdxs:
        outputs[str(outputIdx)] = app.state.interpreter.get_tensor(outputIdx)
    print(outputs) # Return to the API caller


if __name__ == "__main__":
    import numpy as np

    # 1. First the model is uploaded, the input details are returned
    # `Upload model` request saves to app state and initializes interpreter:
    print("`Upload model` request")
    uploadModel()

    # 2. For each tested input, the image is uploaded and result is returned
    inputs = {} # POST data
    for x in app.state.detailsInput:
        inputs[str(x['index'])] = np.random.randint(low=0, high=256, size=x['shape'], dtype=np.uint8)
    # `infer` request:
    print("`infer` request")
    infer(inputs)

    