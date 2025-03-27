from typing import Literal

from datasets import Dataset as hfDataset
from multiprocess.pool import Pool
from PIL import Image


def resizeImages(dataset: hfDataset, columns: dict[str, tuple[tuple[int, int], Literal["RGB"]]]):
    def transformsWrapper(key: str, targetSize: tuple[int, int], mode: str):
        def transforms(item):
            image: Image.Image = item[key]
            item[key] = image.convert(mode).resize(targetSize, resample=Image.Resampling.LANCZOS)
            # item[key] = image.resize(targetSize, resample=Image.Resampling.LANCZOS)
            return item

        return transforms

    transformedDataset = dataset
    for columnName, (targetSize, mode) in columns.items():
        transformFunc = transformsWrapper(key=columnName, targetSize=targetSize, mode=mode)
        transformedDataset = transformedDataset.map(transformFunc, num_proc=Pool()._processes)
    return transformedDataset


def resize_images_and_bboxes(dataset: hfDataset, columns: dict[str, tuple[tuple[int, int], Literal["RGB"]]]):
    def transformsWrapper(imgKey: str, bboxKey: str, targetSize: tuple[int, int], mode: str):
        def transforms(item):
            image: Image.Image = item[imgKey]
            W_orig, H_orig = image.size
            item[imgKey] = image.convert(mode).resize(targetSize, resample=Image.Resampling.LANCZOS)

            item["objects"]["bbox"] = resize_bboxes(
                item["objects"]["bbox"], (W_orig, H_orig), targetSize
            )  # TODO: key?????
            return item

        return transforms

    transformedDataset = dataset
    for columnName, (targetSize, mode) in columns.items():
        transformFunc = transformsWrapper(
            imgKey=columnName,
            bboxKey="",  # TODO
            targetSize=targetSize,
            mode=mode,
        )
        transformedDataset = transformedDataset.map(transformFunc, num_proc=Pool()._processes)
    return transformedDataset


def resize_bboxes(bboxes: list[list[float]], orig_size: tuple[int, int], new_size: tuple[int, int]):
    W_orig, H_orig = orig_size
    W_new, H_new = new_size

    scale_x = W_new / W_orig
    scale_y = H_new / H_orig

    resized_bboxes: list[list[float]] = []
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        x_min_new = x_min * scale_x
        y_min_new = y_min * scale_y
        width_new = width * scale_x
        height_new = height * scale_y
        resized_bboxes.append([x_min_new, y_min_new, width_new, height_new])

    return resized_bboxes
