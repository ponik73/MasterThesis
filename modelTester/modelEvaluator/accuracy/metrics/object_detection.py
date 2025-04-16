import numpy as np
import torch
from torchmetrics.detection.iou import IntersectionOverUnion as TorchMetricsIntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from typing import Literal

from .metric import Metric

class IntersectionOverUnion(Metric):
    def __init__(self, box_format: Literal['xyxy', 'xywh', 'cxcywh']):
        self.torch_metric = TorchMetricsIntersectionOverUnion(box_format=box_format)
    def name(self):
        return "intersection-over-union"
    def compute(self):
        return self.torch_metric.compute()['iou']

class MeanAveragePrecisionMacro(Metric):
    def __init__(self, box_format: Literal['xyxy', 'xywh', 'cxcywh']):
        self.torch_metric = MeanAveragePrecision(box_format=box_format, average='macro')

    def name(self):
        return "mean-average-precision-macro"
    
    def compute(self):
        return float(self.torch_metric.compute()['map'])