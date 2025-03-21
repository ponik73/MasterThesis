import numpy as np
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_recall, multiclass_f1_score, multiclass_precision, multiclass_specificity
from torchmetrics.classification import MulticlassAccuracy

from .metric import Metric
    
class AccuracyMicro(Metric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.torch_metric = MulticlassAccuracy(num_classes=num_classes, average='micro')
    
    def name(self):
        return "accuracy-micro"
    
    # def compute(self, predictions, targets, **kwargs):
    #     num_classes = kwargs.get("num_classes", None)
    #     metric_value = multiclass_accuracy(
    #         preds=predictions,
    #         target=targets,
    #         num_classes=num_classes,
    #         average="micro"
    #     )
    #     return float(metric_value)

    # def compute(self, predictions, targets):
    #     return np.mean(predictions == targets)

# class AccuracyMacro(Metric):


#     def name(self):
#         return "accuracy-macro"
    
#     def compute(self, predictions, targets, **kwargs):
#         num_classes = kwargs.get("num_classes", None)
#         metric_value = multiclass_accuracy(
#             preds=predictions,
#             target=targets,
#             num_classes=num_classes,
#             average="macro"
#         )
#         return float(metric_value)
#     # def compute(self, predictions, targets):
#     #     unique_classes = np.unique(targets)
#     #     accuracies = []
#     #     for cls in unique_classes:
#     #         correct = np.sum((predictions == cls) & (targets == cls))
#     #         total = np.sum(predictions == cls)
#     #         accuracy = correct / total if total > 0 else 0.0
#     #         accuracies.append(accuracy)
#     #     return np.mean(accuracies)

# class RecallMacro(Metric):


#     def name(self):
#         return "recall-macro"
    
#     def compute(self, predictions, targets, **kwargs):
#         num_classes = kwargs.get("num_classes", None)
#         metric_value = multiclass_recall(
#             preds=predictions,
#             target=targets,
#             num_classes=num_classes,
#             average="macro"
#         )
#         return float(metric_value)
#     # def compute(self, predictions, targets):
#     #     unique_classes = np.unique(targets)
#     #     recalls = []
        
#     #     for cls in unique_classes:
#     #         tp = np.sum((targets == cls) & (predictions == cls))
#     #         fn = np.sum((targets == cls) & (predictions != cls))
#     #         recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     #         recalls.append(recall)
        
#     #     return np.mean(recalls)
    
# class RecallMicro(Metric):

    
#     def name(self):
#         return "recall-micro"
    
#     def compute(self, predictions, targets, **kwargs):
#         num_classes = kwargs.get("num_classes", None)
#         metric_value = multiclass_recall(
#             preds=predictions,
#             target=targets,
#             num_classes=num_classes,
#             average="micro"
#         )
#         return float(metric_value)
#     # def compute(self, predictions, targets):
#     #     tp = np.sum(targets == predictions)
#     #     fn = np.sum(targets != predictions)
#     #     return tp / (tp + fn) if (tp + fn) > 0 else 0.0


# class F1Macro(Metric):
    
#     def name(self):
#         return "f1-macro"
    
#     def compute(self, predictions, targets, **kwargs):
#         num_classes = kwargs.get("num_classes", None)
#         metric_value = multiclass_f1_score(
#             preds=predictions,
#             target=targets,
#             num_classes=num_classes,
#             average="macro"
#         )
#         return float(metric_value)
#     # def compute(self, predictions, targets):
        
#     #     unique_classes = np.unique(targets)
#     #     f1_scores = []
        
#     #     for cls in unique_classes:
#     #         tp = np.sum((predictions == cls) & (targets == cls))
#     #         fn = np.sum((predictions != cls) & (targets == cls))
#     #         fp = np.sum((predictions == cls) & (targets != cls))
            
#     #         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     #         recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     #         f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
#     #         f1_scores.append(f1)
        
#     #     return np.mean(f1_scores) if len(f1_scores) > 0 else 0.0

# class F1Micro(Metric):
    
#     def name(self):
#         return "f1-micro"
    
#     def compute(self, predictions, targets, **kwargs):
#         num_classes = kwargs.get("num_classes", None)
#         metric_value = multiclass_f1_score(
#             preds=predictions,
#             target=targets,
#             num_classes=num_classes,
#             average="macro"
#         )
#         return float(metric_value)
#     # def compute(self, predictions, targets):
#     #     tp = np.sum(targets == predictions)
#     #     fn = np.sum((targets != predictions) & (targets == 1))
#     #     fp = np.sum((targets != predictions) & (targets == 0))
        
#     #     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     #     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
#     #     return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


# # TODO: Precision
# class PrecisionMacro(Metric):
    
#     def name(self):
#         return "precision-macro"
    
#     def compute(self, predictions, targets, **kwargs):
#         num_classes = kwargs.get("num_classes", None)
#         metric_value = multiclass_precision(
#             preds=predictions,
#             target=targets,
#             num_classes=num_classes,
#             average="macro"
#         )
#         return float(metric_value)
#     # def compute(self, predictions, targets):
#     #     unique_classes = np.unique(targets)
#     #     precision_scores = []
        
#     #     for cls in unique_classes:
#     #         tp = np.sum((predictions == cls) & (targets == cls))
#     #         fp = np.sum((predictions == cls) & (targets != cls))
            
#     #         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     #         precision_scores.append(precision)
        
#     #     return np.mean(precision_scores) if len(precision_scores) > 0 else 0.0

# class PrecisionMicro(Metric):
    
#     def name(self):
#         return "precision-micro"
    
#     def compute(self, predictions, targets, **kwargs):
#         num_classes = kwargs.get("num_classes", None)
#         metric_value = multiclass_precision(
#             preds=predictions,
#             target=targets,
#             num_classes=num_classes,
#             average="micro"
#         )
#         return float(metric_value)
#     # def compute(self, predictions, targets):
#     #     tp = np.sum(targets == predictions)
#     #     fp = np.sum((targets != predictions) & (predictions == 1))

#     #     return tp / (tp + fp) if (tp + fp) > 0 else 0.0


# class SpecificityMacro(Metric):
    
#     def name(self):
#         return "specificity-macro"
    
#     def compute(self, predictions, targets, **kwargs):
#         num_classes = kwargs.get("num_classes", None)
#         metric_value = multiclass_specificity(
#             preds=predictions,
#             target=targets,
#             num_classes=num_classes,
#             average="macro"
#         )
#         return float(metric_value)
#     # def compute(self, predictions, targets):
#     #     unique_classes = np.unique(targets)
#     #     specificity_scores = []
        
#     #     for cls in unique_classes:
#     #         tn = np.sum((predictions != cls) & (targets != cls))
#     #         fp = np.sum((predictions == cls) & (targets != cls))
            
#     #         specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#     #         specificity_scores.append(specificity)
        
#     #     return np.mean(specificity_scores) if len(specificity_scores) > 0 else 0.0
    

# class SpecificityMicro(Metric):
    
#     def name(self):
#         return "specificity-micro"
    
#     def compute(self, predictions, targets, **kwargs):
#         num_classes = kwargs.get("num_classes", None)
#         metric_value = multiclass_specificity(
#             preds=predictions,
#             target=targets,
#             num_classes=num_classes,
#             average="micro"
#         )
#         return float(metric_value)
#     # def compute(self, predictions, targets):
#     #     tn = np.sum((predictions == 0) & (targets == 0))
#     #     fp = np.sum((predictions == 1) & (targets == 0))

#     #     return tn / (tn + fp) if (tn + fp) > 0 else 0.0
