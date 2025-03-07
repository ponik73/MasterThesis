import numpy as np
from typing import Set

from .metric import Metric
    
class AccuracyMicro(Metric):
    
    def name(self):
        return "accuracy-micro"
    

    def compute(self, predictions, targets):
        return np.mean(predictions == targets)

class AccuracyMacro(Metric):


    def name(self):
        return "accuracy-macro"
    

    def compute(self, predictions, targets):
        unique_classes = np.unique(targets)
        accuracies = []
        for cls in unique_classes:
            correct = np.sum((predictions == cls) & (targets == cls))
            total = np.sum(predictions == cls)
            accuracy = correct / total if total > 0 else 0.0
            accuracies.append(accuracy)
        return np.mean(accuracies)

class RecallMacro(Metric):


    def name(self):
        return "recall-macro"
    
    
    def compute(self, predictions, targets):
        unique_classes = np.unique(targets)
        recalls = []
        
        for cls in unique_classes:
            tp = np.sum((targets == cls) & (predictions == cls))
            fn = np.sum((targets == cls) & (predictions != cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)
        
        return np.mean(recalls)
    
class RecallMicro(Metric):

    
    def name(self):
        return "recall-micro"
    
    
    def compute(self, predictions, targets):
        tp = np.sum(targets == predictions)
        fn = np.sum(targets != predictions)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# TODO: F1 score

# TODO: Precision

# TODO: Specificity