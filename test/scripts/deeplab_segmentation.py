import numpy as np


def align_target_predicted(predictions, targets):
    output_idx = 168
    predictions = [x[output_idx][0] for x in predictions]
    # predictions = np.array([x[0] for x in predictions])
    # predictions = np.squeeze(predictions, axis=1)
    num_classes = predictions[0].shape[-1]
    # predictions = np.argmax(predictions, axis=-1)
    predictions = [np.argmax(x, axis=-1) for x in predictions]
    targets[targets == 255] = 0
    predictions = np.array(predictions).astype("int64")
    targets = targets.astype("int64")
    return (predictions, targets, num_classes, "index")
