from abc import ABC, abstractmethod
from typing import List, Type, Callable, Any, Dict
from pathlib import Path
from types import ModuleType
from importlib.util import spec_from_file_location, module_from_spec
import inspect

from user_script.exceptions import ScriptLoadingError, RequiredFunctionNotPresentError, RequiredFunctionInvalidParameters

class RequiredFunction(ABC):
    def __init__(self):
        self.loaded_callable : Callable | None = None
    
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def param_names(self) -> List[str]:
        pass
    
    def _create_params_dict(self) -> Dict[str, Any]:
        return dict.fromkeys(self.param_names())
    
    @abstractmethod
    def call(self):
        pass
    

class UserScript(ABC):
    def __init__(self, script_path: Path):
        self.script_path = script_path
        self.module : ModuleType | None = None
        self.required_functions : List[RequiredFunction] = []

    def get_function_by_name(self, name:str) -> RequiredFunction | None:
        for f in self.required_functions:
            if f.name() == name:
                return f
        return None

    def load(self):
        spec = spec_from_file_location("user_script", self.script_path)
        self.module = module_from_spec(spec)
        spec.loader.exec_module(self.module)
        if self.module is None:
            raise ScriptLoadingError(script_path=self.script_path)
        
        self._validate()
    
    def _validate(self) -> bool:
        for f in self.required_functions:
            # Module contains object with the required name:
            if not hasattr(self.module, f.name()):
                raise RequiredFunctionNotPresentError(script_path=self.script_path, function_name=f.name())
            module_f = getattr(self.module, f.name())
            
            # The object is callable:
            if not callable(module_f):
                raise RequiredFunctionNotPresentError(script_path=self.script_path, function_name=f.name())
            module_f_signature = inspect.signature(module_f)
            
            # The callable parameters match the required parameters:
            module_f_param_names = list(module_f_signature.parameters.keys())
            if len(module_f_param_names) != len(f.param_names()) or set(module_f_param_names) != set(f.param_names()):
                raise RequiredFunctionInvalidParameters(script_path=self.script_path, function_name=f.name())
            
            f.loaded_callable = module_f
    




