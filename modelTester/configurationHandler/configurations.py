supportedModelHubs = [
        "hf",
        "kaggle"
    ]

# Classes representing configuration file objects:
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

        # Input shape must be a list of integers:
        if not isinstance(input_shape, list) or set([isinstance(x, int) for x in input_shape]) != {True}:
            raise ValueError("Input shape must be a list of integers")
        self.inputShape = input_shape

        # Check if given task is supported:
        if task not in self.supportedTasks:
            raise ValueError(f'Task "{task}" is not supported. Supported tasks are: {self.supportedTasks}')
        self.task = task

        # Check if given model hub is supported:
        if platform not in supportedModelHubs:
            raise ValueError(f'Model hub "{platform}" is not supported. Supported platforms are: {supportedModelHubs}')
        self.platform = platform
        
class Dataset():
    attributes = ["name", "platform", "uri"]
    distinctAttributes = ["name"]

    def __init__(self, name: str, platform: str, uri: str):
        self.name = name
        self.uri = uri

        # Check if given model hub is supported:
        if platform not in supportedModelHubs:
            raise ValueError(f'Model hub "{platform}" is not supported. Supported platforms are: {supportedModelHubs}')
        self.platform = platform
        
class Run():
    attributes = ["device", "model", "dataset"]
    distinctAttributes = []
    def __init__(self, device: str, model: str, dataset: str):
        self.deviceName = device
        self.modelName = model
        self.datasetName = dataset
