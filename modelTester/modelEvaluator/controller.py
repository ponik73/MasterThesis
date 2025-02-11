from configurationHandler.configurations import Device, Run, Model, Dataset
from .device.controller import DeviceController

import asyncio, asyncssh

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

        # If device is available, create controller and save device's fingerprint:
        self.deviceControllers : list[DeviceController] = []
        for x in devices:
            controller = self._isDeviceAvailable(x)
            if controller:
                print(f'Device {x.name} discovered')
                self.deviceControllers.append(controller)
            else:
                print(f'Device {x.name} could not be discovered')
        # Names of available devices, models, and datasets:
        namesAvailable = lambda lst : [x.name for x in lst]
        # Filter runs based on available models/datasets:
        self.runs = [x for x in runs if (x.deviceName in namesAvailable([x.device for x in self.deviceControllers]) and x.modelName in namesAvailable(self.models) and x.datasetName in namesAvailable(self.datasets))]
    
    def _isDeviceAvailable(self, dev: Device) -> DeviceController | None:
        deviceController = DeviceController(dev)
        try:
            # asyncio.get_event_loop().run_until_complete(run_client())
            available = asyncio.get_event_loop().run_until_complete(deviceController.getFingerprint())
            if available:
                return deviceController
            return None
        except (OSError, asyncssh.Error) as exc:
            print('SSH connection failed: ' + str(exc))
            return None
    
    def finished(self):
        return len(self.runs) == 0
    
    def executeRun(self):
        try:
            run = self.runs.pop()
            
            print(f'RUN:\n\tdevice:\t{run.deviceName}\n\tmodel:\t{run.modelName}\n\tdataset:\t{run.datasetName}')
            
            deviceController = [x for x in self.deviceControllers if x.device.name == run.deviceName][0]
            model = [x for x in self.models if x.name == run.modelName][0]
            
            print("Uploading model ...")
            modelRemotePath = asyncio.get_event_loop().run_until_complete(deviceController.uploadModel(model.localPath))
            # TODO: latency
            print("Latency assessment ...")
            latency = asyncio.get_event_loop().run_until_complete(deviceController.evaluateLatency(modelRemotePath))
            # TODO: dataset??
            # TODO: accuracy??
            # TODO: write to the DB
            print(latency)
        except Exception as ex:
            print(str(ex))
            return None