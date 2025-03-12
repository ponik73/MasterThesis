import pickle
import codecs
from typing import Any    

def encodeData(data: Any) -> bytes:
    return codecs.encode(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL), "base64")

def decodeData(data: bytes) -> Any:
    try:
        return pickle.loads(codecs.decode(data, "base64"))
    except Exception as e:
        raise Exception(f"Error during decoding data. Reason: '{e}'.") from e