from typing import Dict, List, Any
from numpy.typing import NDArray
import numpy as np
from torchmetrics.functional.detection.iou import intersection_over_union
from torchmetrics.detection.iou import IntersectionOverUnion
from torch.utils.data import DataLoader

from ..accuracyEvaluator import AccuracyEvaluator
from .utils import resize_images_and_bboxes
from modelEvaluator.api.utils import encodeData, decodeData

class ObjectDetectionAccuracyEvaluator(AccuracyEvaluator):
    """Class for evaluating single input and single output computer vision models"""
    
    def __init__(self, model, deviceInterface, dataset):
        super().__init__(model, deviceInterface, dataset)

        pass
        # TODO

    def evaluate(self):
        pass
        # TODO:
        