import numpy as np
from pydantic import BaseModel


class _InputInfoTflite(BaseModel):
    index: int
    shape: list[int]
    dtype: str

    @classmethod
    def from_dict(cls, data: dict) -> "_InputInfoTflite":
        return cls(index=data["index"], shape=data["shape"], dtype=np.dtype(data["dtype"]).name)

    def to_dict(self) -> dict[str, int | list[int] | str]:
        return {"index": self.index, "shape": self.shape, "dtype": self.dtype}


class InputInfoTfliteOutput(BaseModel):
    """API output class for tflite model input info."""

    inputs: list[_InputInfoTflite]

    @classmethod
    def from_list(cls, data_list: list[dict]) -> "InputInfoTfliteOutput":
        """Initiate class instance from list.

        :param data_list: List of dictionaries containing node details
        :return: InputInfoTfliteOutput instance
        """
        return cls(inputs=[_InputInfoTflite.from_dict(x) for x in data_list])
