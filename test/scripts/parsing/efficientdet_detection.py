import numpy as np
from numpy.typing import NDArray
import cv2

def postprocess_raw_detections(output):
    boxes = []
    scores = []
    class_ids = []

    for pred in output:
        pred = np.transpose(pred)
        for box in pred:
                x, y, w, h = box[:4]
                x1 = x - w / 2
                y1 = y - h / 2
                boxes.append([x1, y1, w, h])
                idx = np.argmax(box[4:])
                scores.append(box[idx + 4])
                class_ids.append(idx)
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)
    
    d = {
        "boxes": [],
        "scores": [],
        "labels": []
    }
    for i in indices:
        score = scores[i]
        if score > 0.25:
            d["boxes"].append(boxes[i])
            d["scores"].append(score)
            d["labels"].append(class_ids[i] + 1)

    return d

def align_target_predicted(predictions, targets):
    predictions = np.array([x[517].squeeze() for x in predictions])
    
    zero_point = -128
    scale = 0.005322090815752745
    predictions = (predictions.astype(np.float32) - zero_point) * scale
    predictions[:, [0, 2]] *= 320
    predictions[:, [1, 3]] *= 320

    _predictions = []
    for output in predictions:
        output = np.array([output])
        processed = postprocess_raw_detections(output)
        _predictions.append(processed)

    boxes, labels = targets
    _targets = []
    for i, b in enumerate(boxes):
        _targets.append({
            "boxes": b,
            "labels": labels[i]
        })

    return _predictions, _targets, 'xywh'