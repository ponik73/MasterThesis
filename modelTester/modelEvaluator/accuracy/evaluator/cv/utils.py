from datasets import features, formatting
from PIL import Image
from datasets import Dataset as hfDataset
from typing import List, Dict, Tuple, Literal, Callable, Any

def resizeImages(
        dataset: hfDataset,
        columns : Dict[str, Tuple[Tuple[int, int], Literal["RGB"]]]
):
    def transformsWrapper(key: str, targetSize: Tuple[int, int], mode: str):
        def transforms(item):
            image : Image.Image = item[key]
            item[key] = image.convert(mode).resize(targetSize, resample=Image.Resampling.LANCZOS)
            return item
        return transforms
    
    transformedDataset = dataset
    for columnName, (targetSize, mode) in columns.items():
        transformFunc = transformsWrapper(
            key=columnName,
            targetSize=targetSize,
            mode=mode
        )
        transformedDataset = transformedDataset.map(transformFunc)
    return transformedDataset