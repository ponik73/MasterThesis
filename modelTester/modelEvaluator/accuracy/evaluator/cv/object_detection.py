from typing import Dict, List, Any
from numpy.typing import NDArray
import numpy as np
from torchmetrics.functional.detection.iou import intersection_over_union
from torchmetrics.detection.iou import IntersectionOverUnion
from torch.utils.data import DataLoader

from ..accuracyEvaluator import AccuracyEvaluator
from .utils import resize_images_and_bboxes
from modelEvaluator.api.utils import encodeData, decodeData

def iou(box1, box2) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def non_maximum_suppression(detections: List[List[float]], iou_threshold=0.5) -> List[List[float]]:
    """Applies Non-Maximum Suppression (NMS) to remove overlapping bounding boxes."""
    detections.sort(key=lambda x: x[4], reverse=True)  # Sort by confidence score
    final_detections = []
    while detections:
        best_box = detections.pop(0)
        final_detections.append(best_box)
        detections = [box for box in detections if iou(best_box, box) < iou_threshold]
    return final_detections

def aaa(pred: Dict[int, NDArray[Any]]):
    detections = []
    boxes_id, classes_id, scores_id = 167, 168, 169
    boxes = pred[boxes_id].reshape(-1,4)
    scores = pred[scores_id].flatten()
    classes = pred[classes_id].flatten()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(float, boxes[i])
        score = scores[i]
        class_id = int(classes[i])
        if score > 0.25:  # Filter low-confidence boxes
            detections.append([x1, y1, x2, y2, score, class_id])
    return detections

# def process_output(pred: Dict[int, NDArray[Any]]):
#     detections = []
#     bbox_node, score_node, class_node = 387, 388, 386 # TODO: config
#     bboxes = pred[bbox_node].reshape(-1, 4)
#     scores = pred[score_node].flatten()
#     classes = pred[class_node].flatten()
#     for i in range(len(bboxes)):
#         x1, y1, x2, y2 = map(float, bboxes[i])
#         if np.issubdtype(scores[i], np.integer):
#             score = float(scores[i] / 255.0)
#         elif np.issubdtype(scores[i], np.floating):
#             score = scores[i]
#         class_id = int(class_ids[i])
#         if score > 0.25:  # Filter low-confidence boxes
#             detections.append([x1, y1, x2, y2, score, class_id])
#     detections = non_maximum_suppression(detections)
#     prediction = {}
#     prediction["boxes"] = np.array([x[:4] for x in detections])
#     prediction["labels"] = np.array([x[-1] for x in detections])
#     return prediction

# def somef(i):

# aa = process_output(preds[0])

class ObjectDetectionAccuracyEvaluator(AccuracyEvaluator):
    """Class for evaluating single input and single output computer vision models"""
    
    def __init__(self, model, deviceInterface, dataset):
        super().__init__(model, deviceInterface, dataset)

        # # Get the name of the feature which contains images:
        self.imageKey : str | None = None
        self.objectsKey : str | None = None

        self.imageKey = "image"
        self.objectsKey = "objects"

        # for key, value in dataset.dataset.features.items():
        #     if isinstance(value, features.image.Image):
        #         self.imageKey = key
        #     if isinstance(value, features.features.ClassLabel):
        #         self.labelKey = key
        # self.labelKey = 'objects'
        # if not self.imageKey or not self.labelKey:
        #     raise ValueError(f"Dataset {self.dataset.name} does not contain features in required format.")

    def _predict(self):
        return
        inputShape : List[int] = self.modelInputs[0]["shape"]
        targetSize = (inputShape[1], inputShape[2])
        transformedDataset = resize_images_and_bboxes(
            dataset=self.dataset.dataset,
            columns={
                self.imageKey: (targetSize, "RGB")
            }
        )

        # Create data loader and batches:
        inputDtype : str = self.modelInputs[0]["dtype"]
        def collate_fn(batch):
            # images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            
            if np.issubdtype(inputDtype, np.integer):
                images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            elif np.issubdtype(inputDtype, np.floating):
                images = np.array([np.array(x[self.imageKey]) / 255.0 for x in batch]).astype(inputDtype)
                images = np.array([x[self.imageKey] for x in batch]).astype(inputDtype)
            else:
                raise ValueError("TODO")
            
            objects = np.array([x[self.objectsKey] for x in batch])
            return {
                self.imageKey: images,
                self.objectsKey: objects
            }
        dataloader = DataLoader(transformedDataset, shuffle=True, batch_size=self.BATCH_SIZE, collate_fn=collate_fn, drop_last=True)
        
        inputIdx : int = self.modelInputs[0]["index"]
        # Inference for each batch:
        for batch in dataloader:
            images : np.typing.NDArray[Any] = batch[self.imageKey]
            labels : np.typing.NDArray[Any] = batch[self.objectsKey]
            
            if self.targets is None:
                self.targets = np.array(labels)
            else:
                # self.targets = np.concatenate((self.targets, [labels]))
                self.targets = np.append(self.targets, labels, axis=0)
            
            mappingInputData : List[Dict[int, np.typing.NDArray[Any]]] = [{inputIdx: x} for x in images]

            # Inference:
            encodedOutputs = self.deviceInterface.predict(
                modelCustomName=self.model.name,
                modelFramework=self.model.framework,
                encodedBatch=encodeData(mappingInputData)
            )
            if not encodedOutputs:
                raise Exception(f'[{self.deviceInterface.name}] Error during on device inference.')
            
            # Decode inference result and append it to the predictions:
            modelOutputs : List[Dict[int, np.typing.NDArray[Any]]] = decodeData(encodedOutputs)
            if self.predictions is None:
                self.predictions = np.array(modelOutputs)
            else:
                self.predictions = np.append(self.predictions, modelOutputs, axis=0)
            break
    
    def _calculateMetrics(self):
        return
        np.save("preds1.npy" ,self.predictions)
        np.save("target1.npy" ,self.targets)
        
        pass