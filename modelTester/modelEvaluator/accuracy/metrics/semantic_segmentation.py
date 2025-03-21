import numpy as np
import torch
from torchmetrics.segmentation import MeanIoU
from torchmetrics.functional.segmentation import mean_iou
from typing import Literal

from .metric import Metric

class MeanIntersectionOverUnion(Metric):
    def __init__(self, num_classes: int, input_format: Literal['one-hot', 'index']):
        self.torch_metric = MeanIoU(num_classes=num_classes, input_format=input_format)

    def name(self):
        return "mean-intersection-over-union"
    