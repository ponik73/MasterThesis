from user_script.script import RequiredFunction, UserScript
from numpy.typing import NDArray
from typing import Any

class AlignTargetPredictedFunction(RequiredFunction):
    def name(self):
        return "align_target_predicted"
    
    def param_names(self):
        return ["predictions", "targets"]
    
    def call(self, predictions: NDArray[Any], targets: NDArray[Any]):
        kwargs = self._create_params_dict()
        kwargs["predictions"] = predictions
        kwargs["targets"] = targets

        return self.loaded_callable(**kwargs)

class AlignTargetPredictedScript(UserScript):
    def __init__(self, script_path):
        super().__init__(script_path)
        self.required_functions.append(AlignTargetPredictedFunction())