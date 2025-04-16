from typing import Literal
from collections.abc import Callable
from datasets import Dataset as hfDataset
from multiprocess.pool import Pool
from PIL import Image


def resize_images(dataset: hfDataset, columns: dict[str, tuple[tuple[int, int], Literal["RGB"]]]) -> hfDataset:
    """
    Resize images in a Hugging Face dataset to specified dimensions and color modes.
    """
    def transforms_wrapper(key: str, target_size: tuple[int, int], mode: str) -> Callable:
        def transforms(item):
            image: Image.Image = item[key]
            item[key] = image.convert(mode).resize(target_size, resample=Image.Resampling.LANCZOS)
            # item[key] = image.resize(target_size, resample=Image.Resampling.LANCZOS)
            return item

        return transforms

    transformed_dataset = dataset
    for column_name, (target_size, mode) in columns.items():
        transform_func = transforms_wrapper(key=column_name, target_size=target_size, mode=mode)
        transformed_dataset = transformed_dataset.map(transform_func, num_proc=Pool()._processes)
    return transformed_dataset


def _resize_bboxes(bboxes: list[list[float]], orig_size: tuple[int, int], new_size: tuple[int, int])  -> list[list[float]]:
    w_orig, h_orig = orig_size
    w_new, h_new = new_size
    scale_x = w_new / w_orig
    scale_y = h_new / h_orig
    resized_bboxes: list[list[float]] = []
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        x_min_new = x_min * scale_x
        y_min_new = y_min * scale_y
        width_new = width * scale_x
        height_new = height * scale_y
        resized_bboxes.append([x_min_new, y_min_new, width_new, height_new])
    return resized_bboxes


def resize_bboxes(
    dataset: hfDataset,
    target_size: tuple[int, int],
    img_key: str,
    bbox_key: list[str],
) -> hfDataset:
    """
    Resize bounding boxes in a Hugging Face dataset to reflect new image dimensions.
    """
    def transforms(item):
        image: Image.Image = item[img_key]
        w_orig, h_orig = image.width, image.height
        bboxes: list[list[float]] = nested_read(item, bbox_key)
        resized = _resize_bboxes(bboxes, (w_orig, h_orig), target_size)
        _nested_write(item, resized, bbox_key)
        return item

    return dataset.map(transforms, num_proc=Pool()._processes)


def nested_read(row, keys: list):
    """
    Recursively retrieve a nested value from a dictionary using a list of keys.
    """
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
