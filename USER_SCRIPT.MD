# User script specification

User-defined scripts are an essential component of the evaluation pipeline. Their primary role is to provide custom logic for post-processing the raw outputs of the evaluated model and aligning them with the corresponding ground truth data. Each script is specific to a particular (model + dataset) pair, and its path is provided in the configuration file for each run.

All user scripts must implement a function named:
```
align_target_predicted(predictions, targets)
```

- predictions: A `list[dict[int, np.NDArray]]`, representing the raw outputs of the evaluated model for each input in the batch.
- targets: A list, representing the corresponding ground truth annotations, extracted from the test dataset using the key(s) defined in the configuration file.
- Return value: The function must return a tuple of task-specific outputs described below.

## Image classification

- `targets`: A list of ground truth class labels (e.g., [3, 7, 2, ...]).

The function must return:

1. `predictions`: A `np.NDArray[np.Number]` containing predicted class indices.
2. `targets`: A `np.NDArray[np.Number]` of ground truth labels aligned with predictions.
3. `num_classes`: An integer - the total number of classes. The user is expected to know this based on the dataset.

## Semantic segmentation

- `targets`: A list of ground truth segmentation masks, each represented as a *NumPy array*.

The function must return:

1. `predictions`: A `np.NDArray[np.NDArray]` containing predicted segmentation masks.
2. `targets`: A `np.NDArray[np.NDArray]` of ground truth masks aligned with predictions.
3. `num_classes`: An integer - the number of segmentation classes (user-defined).
4. `input_format`: A string indicating the format of the segmentation masks. Accepted values:
    - 'one-hot': For one-hot encoded outputs.
    - 'index': For class index tensors.

## Object detection

- `targets`: A list of tuples (boxes, labels), where both are `np.NDArray`. Boxes contain ground truth bounding boxes, and labels contain corresponding class labels.

The function must return:

1. `predictions`: A list[dict[str, list]], where each dictionary must include:
    - "boxes": List of predicted bounding boxes.
    - "scores": List of confidence scores.
    - "labels": List of predicted class labels.
2. `targets`: Ground truth annotations aligned with predictions.
3. `box_format`: A string indicating the format of the bounding boxes. Supported values:
    - 'xyxy': Coordinates of top-left (x1, y1) and bottom-right (x2, y2) corners.
    - 'xywh': Top-left corner (x1, y1), width (w), and height (h).
    - 'cxcywh': Center point (cx, cy), width (w), and height (h).