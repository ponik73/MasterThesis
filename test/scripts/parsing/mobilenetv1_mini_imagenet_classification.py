from typing import List, Tuple, Any
import numpy as np
from numpy.typing import NDArray

# def align_target_predicted(predictions: NDArray[Any], targets: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
def align_target_predicted(predictions, targets):
    outputIdx = list(predictions.item(0).keys())[0]
    predictions = np.array([x[outputIdx] for x in predictions])
    num_classes = predictions[0].shape[-1]
    predictions = np.array([x.argmax().item() for x in predictions])
    predictions = np.array([x - 1 for x in predictions])
    return (predictions, targets, num_classes)