import numpy as np
import torch
from torchmetrics.segmentation import MeanIoU, DiceScore, HausdorffDistance
from typing import Literal

from .metric import Metric

class MeanIntersectionOverUnion(Metric):
    def __init__(self, num_classes: int, input_format: Literal['one-hot', 'index']):
        self.torch_metric = MeanIoU(num_classes=num_classes, input_format=input_format)

    def name(self):
        return "mean-intersection-over-union"

class DiceScoreMicro(Metric):
    def __init__(self, num_classes: int, input_format: Literal['one-hot', 'index']):
        super().__init__()
        self.torch_metric = DiceScore(num_classes=num_classes, input_format=input_format, average='micro')
    
    def name(self):
        return "dice-score-micro"
    
class DiceScoreMacro(Metric):
    def __init__(self, num_classes: int, input_format: Literal['one-hot', 'index']):
        super().__init__()
        self.torch_metric = DiceScore(num_classes=num_classes, input_format=input_format, average='micro')
    
    def name(self):
        return "dice-score-macro"

class HausdorffDistanceEuclidean(Metric):
    def __init__(self, num_classes: int, input_format: Literal['one-hot', 'index']):
        super().__init__()
        self.torch_metric = HausdorffDistance(num_classes=num_classes, input_format=input_format, distance_metric='euclidean')

    def name(self):
        return "hausdorff-distance-euclidean"