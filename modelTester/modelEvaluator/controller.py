from configurationHandler.configurations import Device, Run, Model, Dataset
from typing import List, Tuple
from pydantic_core import Url
from collections import defaultdict
import asyncio

from .device.interface import DeviceInterface
from .pipeline import RunPipeline

class EvaluatorController():
    def __init__(
            self,
            devices: List[Device],
            runs: List[Run]
            ):
        self.models : List[Model] = []
        self.datasets : List[Dataset] = []
        self.runs = runs
        self.devices : List[Device] = []
        self.deviceInteraces : List[DeviceInterface] = []
        self.pipelines : List[RunPipeline] = []

        # Discover devices:
        for dev in devices:
            try:
                # Initialize the device API:
                interface = DeviceInterface(url=Url(dev.uri), name=dev.name)
                # Get device fingerprint:
                dev.fingerprint = interface.getFingerprint()

                self.devices.append(dev)
                self.deviceInteraces.append(interface)

                print(f'Device `{dev.name}` at `{dev.uri}` was discovered.')
            except Exception as e:
                print(f'Unable to discover `{dev.name}` at `{dev.uri}`. Reason: {e}')

    def setModels(self, models: List[Model]):
        # Remove models where localPath is None (not downloaded):
        self.models = [x for x in models if x.localPath]

    def setDatasets(self, datasets: List[Dataset]):
        # Remove datasets where dataset is None (not downloaded):
        self.datasets = [x for x in datasets if x.dataset]

    def createPipelines(self):
        # Names of available devices, models, and datasets:
        namesDevices = {device.name for device in self.devices}
        namesModels = {model.name for model in self.models}
        namesDatasets = {dataset.name for dataset in self.datasets}
        # Filter runs based on available models/datasets:
        self.runs = [
            run for run in self.runs
            if run.deviceName in namesDevices
            and run.modelName in namesModels
            and run.datasetName in namesDatasets
        ]

        # Group runs by device:
        runsByDevice = defaultdict(list)
        for run in self.runs:
            runsByDevice[run.deviceName].append(run)
        runsByDevice = dict(runsByDevice)

        # Create pipeline for each device:
        for deviceName in runsByDevice.keys():
            # Interface for a given device:
            interface = [x for x in self.deviceInteraces if x.name == deviceName][0]

            # Run for a device is defined by model and dataset:
            pairsModelDataset : List[Tuple[Model, Dataset]] = []
            # Create pairs of models and datasets:
            for run in runsByDevice[deviceName]:
                model = [x for x in self.models if x.name == run.modelName][0]
                dataset = [x for x in self.datasets if x.name == run.datasetName][0]
                pairsModelDataset.append((model, dataset))
            # Order pairs by model name:
            pairsModelDataset = sorted(pairsModelDataset, key=lambda pair: pair[0].name)
            
            self.pipelines.append(
                RunPipeline(
                    deviceInterface=interface,
                    pairsModelDataset=pairsModelDataset
                )
            )

    def executePipelines(self):
        for pipeline in self.pipelines:
            pipeline.execute()