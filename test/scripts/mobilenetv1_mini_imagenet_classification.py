import numpy as np


# def align_target_predicted(predictions: NDArray[Any], targets: NDArray[Any]) -> Tuple[NDArray[Any], NDArray[Any]]:
def align_target_predicted(predictions, targets):
    output_idx = next(iter(predictions.item(0).keys()))
    predictions = np.array([x[output_idx] for x in predictions])
    num_classes = predictions[0].shape[-1]
    predictions = np.array([x.argmax().item() for x in predictions])
    predictions = np.array([x - 1 for x in predictions])
    return (predictions, targets, num_classes)
