from datasets import features, formatting
from PIL import Image
from datasets import Dataset as hfDataset
from datasets.arrow_dataset import LazyDict
from typing import List, Dict, Tuple, Literal, Callable, Any
from multiprocess.pool import Pool
import numpy as np

def resizeImages(
        dataset: hfDataset,
        columns : Dict[str, Tuple[Tuple[int, int], Literal["RGB"]]]
):
    def transformsWrapper(key: str, targetSize: Tuple[int, int], mode: str):
        def transforms(item):
            image : Image.Image = item[key]
            item[key] = image.convert(mode).resize(targetSize, resample=Image.Resampling.LANCZOS)
            # item[key] = image.resize(targetSize, resample=Image.Resampling.LANCZOS)
            return item
        return transforms
    
    transformedDataset = dataset
    for columnName, (targetSize, mode) in columns.items():
        transformFunc = transformsWrapper(
            key=columnName,
            targetSize=targetSize,
            mode=mode
        )
        transformedDataset = transformedDataset.map(transformFunc, num_proc=Pool()._processes)
    return transformedDataset

def _resize_bboxes(bboxes: List[List[float]], orig_size: Tuple[int, int], new_size: Tuple[int, int]):
    W_orig, H_orig = orig_size
    W_new, H_new = new_size
    scale_x = W_new / W_orig
    scale_y = H_new / H_orig
    resized_bboxes : List[List[float]] = []
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        x_min_new = x_min * scale_x
        y_min_new = y_min * scale_y
        width_new = width * scale_x
        height_new = height * scale_y
        resized_bboxes.append([x_min_new, y_min_new, width_new, height_new])
    return resized_bboxes

def resize_bboxes(
    dataset,#: hfDataset,
    targetSize: Tuple[int, int],
    imgKey: str,
    bboxKey: list[str]
):
    def transforms(item):
        image : Image.Image = item[imgKey]
        W_orig, H_orig = image.width, image.height
        bboxes : List[List[float]] = nested_read(item, bboxKey)
        resized = _resize_bboxes(bboxes, (W_orig, H_orig), targetSize)
        _nested_write(item, resized, bboxKey)
        return item
    return dataset.map(transforms, num_proc=Pool()._processes)

def nested_read(row, keys: list):
    obj = row[keys[0]]
    if len(keys[1:]) < 1:
        return obj
    return nested_read(obj, keys[1:])

def _nested_write(row, value, keys: list):
    if len(keys) == 1:
        row[keys[0]] = value
        return row
    row[keys[0]] = _nested_write(row[keys[0]], value, keys[1:])
    return row
