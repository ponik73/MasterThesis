supportedPlatforms = [
        "hf",
        "kaggle"
    ]

class Device():
    attributes = ["name", "uri"]
    distinctAttributes = ["name", "uri"]

    def __init__(self, name: str, uri: str):
        self.name = name
        self.uri = uri

class Model():
    attributes = ["name", "task", "input_shape", "platform", "uri"]
    distinctAttributes = ["name", "uri"]

    supportedTasks = [
        "image-classification",
        "object-detection",
        "semantic-segmentation"
        ]
    def __init__(self, name: str, task: str, input_shape, platform: str, uri: str):
        self.name = name
        self.uri = uri

        if not isinstance(input_shape, list) or set([isinstance(x, int) for x in input_shape]) != {True}:
            raise ValueError("Input shape must be a list of integers")
        self.inputShape = input_shape

        if task not in self.supportedTasks:
            raise ValueError(f'Task "{task}" is not supported. Supported tasks are: {self.supportedTasks}')
        self.task = task

        if platform not in supportedPlatforms:
            raise ValueError(f'Platform "{platform}" is not supported. Supported platforms are: {supportedPlatforms}')
        self.platform = platform
        
class Dataset():
    attributes = ["name", "platform", "uri"]
    distinctAttributes = ["name"]

    def __init__(self, name: str, platform: str, uri: str):
        self.name = name
        self.uri = uri

        if platform not in supportedPlatforms:
            raise ValueError(f'Platform "{platform}" is not supported. Supported platforms are: {supportedPlatforms}')
        self.platform = platform
        
class Run():
    attributes = ["device", "model", "dataset"]
    distinctAttributes = []
    def __init__(self, device: str, model: str, dataset: str):
        self.deviceName = device
        self.modelName = model
        self.datasetName = dataset

# Downloader: models, datasets
# Model: (models, datasets) runs, devices

class DownloaderConfiguration():
    def __init__(self, models: list[Model], datasets: list[Dataset]):
        self.models = {
            supportedPlatforms[0]: [m for m in models if m.platform == supportedPlatforms[0]], # Hugging face
            supportedPlatforms[1]: [m for m in models if m.platform == supportedPlatforms[1]]  # Kaggle
        }
        self.datasets = {
            supportedPlatforms[0]: [x for x in datasets if x.platform == supportedPlatforms[0]], # Hugging face
            supportedPlatforms[1]: [x for x in datasets if x.platform == supportedPlatforms[1]]  # Kaggle
        }