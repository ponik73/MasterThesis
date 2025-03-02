from pydantic import BaseModel, Field
from typing import NamedTuple, List, Dict
import numpy as np

class _InputInfoTflite(BaseModel):
    index: int
    shape: List[int]
    dtype: str

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(index=data["index"], shape=data["shape"], dtype=np.dtype(data["dtype"]).name)
    
    def to_dict(self):
        return {"index": self.index, "shape": self.shape, "dtype": self.dtype}

class InputInfoTfliteOutput(BaseModel):
    inputs: List[_InputInfoTflite]

    @classmethod
    def from_list(cls, data_list: List[dict]):
        return cls(inputs=[_InputInfoTflite.from_dict(x) for x in data_list])
    
    def to_list(self):
        return [x.to_dict() for x in self.inputs]
