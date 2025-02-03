from configurationHandler.configurations import Device, Run, Model, Dataset

class EvaluatorController():
    def __init__(
            self,
            devices: list[Device],
            runs: list[Run],
            models: list[Model],
            datasets: list[Dataset]
            ):
        # Remove models where localPath is None (not downloaded):
        self.models = [x for x in models if x.localPath]
        # Remove datasets where dataset is None (not downloaded):
        self.datasets = [x for x in datasets if x.dataset]

        # TODO: check devices if available
        self.devices = devices
        # Names of available devices, models, and datasets:
        namesAvailable = lambda lst : [x.name for x in lst]
        # Filter runs based on available models/datasets:
        self.runs = [x for x in runs if (x.deviceName in namesAvailable(self.devices) and x.modelName in namesAvailable(self.models) and x.datasetName in namesAvailable(self.datasets))]
        
    