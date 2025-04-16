import codecs
import pickle
from typing import Any


def encode_data(data: Any) -> bytes:  # noqa: ANN401
    """Encode given data to bytes."""
    return codecs.encode(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), "base64")


def decode_data(data: bytes) -> Any:  # noqa: ANN401
    """Decode given bytes."""
    return pickle.loads(codecs.decode(data, "base64"))
