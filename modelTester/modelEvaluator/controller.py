from configurationHandler.schemas import Device, Run, Model, Dataset
from typing import List, Tuple, Dict
from pydantic_core import Url
from collections import defaultdict
from pathlib import Path
import asyncio

from modelEvaluator.accuracy.metrics.metric import MetricResult
from modelEvaluator.api.device.interface import DeviceInterface
from modelEvaluator.api.report.interface import ReportInterface
from modelEvaluator.pipeline import RunPipeline
from configuration import getSettings

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
        self.reportInterface = ReportInterface()
        self.run_ids : List[int] = []

        # Discover devices:
        for dev in devices:
            try:
                # Initialize the device API:
                interface = DeviceInterface(device=dev)
                # Get device fingerprint:
                interface.getFingerprint()

                self.devices.append(interface.device)
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
        runsByDevice : Dict[str, List[Run]] = dict(runsByDevice)

        # Create pipeline for each device:
        for deviceName in runsByDevice.keys():
            # Interface for a given device:
            interface = [x for x in self.deviceInteraces if x.device.name == deviceName][0]

            # Run for a device is defined by model and dataset:
            runSettings : List[Tuple[Model, Dataset, Dict[str, str]]] = []
            # Create pairs of models and datasets:
            for run in runsByDevice[deviceName]:
                model = [x for x in self.models if x.name == run.modelName][0]
                dataset = [x for x in self.datasets if x.name == run.datasetName][0]
                runSettings.append((model, dataset, run.required_script_paths))
            # Order pairs by model name:
            runSettings = sorted(runSettings, key=lambda setting: setting[0].name)
            
            self.pipelines.append(
                RunPipeline(
                    deviceInterface=interface,
                    # pairsModelDataset=pairsModelDataset
                    runSettings=runSettings
                )
            )

    def executePipelines(self):
        for pipeline in self.pipelines:
            self._execute_pipeline(pipeline)

    def _execute_pipeline(self, pipeline: RunPipeline):
        device = pipeline.deviceInterface.device
        while not pipeline.empty():
            model, dataset, metrics = pipeline.execute_run()
            run_id = self.reportInterface.register_run(
                device=device,
                model=model,
                dataset=dataset,
                metrics=metrics
            )
            if run_id:
                self.run_ids.append(run_id)
    
    def get_report(self) -> Path | None:
        report = self.reportInterface.generate_report(self.run_ids)
        
        if report:
            report_path_dir = getSettings().REPORT_SAVE_PATH
            report_path_dir.mkdir(parents=True, exist_ok=True)

            file_name = f"report-{'-'.join([str(x) for x in self.run_ids])}.html"
            report_path = report_path_dir / file_name
            
            with open(report_path, "w+") as f:
                f.write(report)

            return report_path
        
        return None