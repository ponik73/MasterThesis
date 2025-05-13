from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from configuration import get_settings
from configurationHandler.schemas import Dataset, Device, Model, Run
from modelEvaluator.api.device.interface import DeviceInterface
from modelEvaluator.api.report.interface import ReportInterface
from modelEvaluator.exceptions import DeviceApiError
from modelEvaluator.pipeline import RunPipeline
from utils import print_success, print_failure


class EvaluatorController:
    """Controller for evaluation pipeline - from device discovery to final reporting."""

    def __init__(self, devices: list[Device], runs: list[Run]) -> None:
        """Initialize the EvaluatorController.

        :param devices: List of available devices to run evaluations on.
        :param runs: List of run configurations (device, model, dataset).
        """
        self.models: list[Model] = []
        self.datasets: list[Dataset] = []
        self.runs = runs
        self.devices: list[Device] = []
        self.deviceInteraces: list[DeviceInterface] = []
        self.pipelines: list[RunPipeline] = []
        self.reportInterface = ReportInterface()
        self.run_ids: list[int] = []

        # Discover devices:
        for dev in devices:
            try:
                # Initialize the device API:
                interface = DeviceInterface(device=dev)
                # Get device fingerprint:
                interface.get_fingerprint()

                self.devices.append(interface.device)
                self.deviceInteraces.append(interface)

                print_success(f"Device `{dev.name}` at `{dev.uri}` was discovered.")
            except DeviceApiError as e:
                print_failure(f"Unable to discover `{dev.name}` at `{dev.uri}`. Reason: {e}")

    def set_models(self, models: list[Model]) -> None:
        """Set the list of models to be evaluated.

        Only models with valid `local_path` will be used.
        """
        # Remove models where local_path is None (not downloaded):
        self.models = [x for x in models if x.local_path]

    def set_datasets(self, datasets: list[Dataset]) -> None:
        """Set the list of datasets to be used for evaluation.

        Only datasets with a loaded `dataset` attribute will be used.
        """
        # Remove datasets where dataset is None (not downloaded):
        self.datasets = [x for x in datasets if x.dataset]

    def create_pipelines(self) -> None:
        """Create evaluation pipelines for each valid run configuration."""
        # Names of available devices, models, and datasets:
        names_devices = {device.name for device in self.devices}
        names_models = {model.name for model in self.models}
        names_datasets = {dataset.name for dataset in self.datasets}
        # Filter runs based on available models/datasets:
        self.runs = [
            run
            for run in self.runs
            if run.device_name in names_devices
            and run.model_name in names_models
            and run.dataset_name in names_datasets
        ]

        # Group runs by device:
        runs_by_device = defaultdict(list)
        for run in self.runs:
            runs_by_device[run.device_name].append(run)
        runs_by_device: dict[str, list[Run]] = dict(runs_by_device)

        # Create pipeline for each device:
        for device_name in runs_by_device:
            # Interface for a given device:
            interface = next((x for x in self.deviceInteraces if x.device.name == device_name), None)

            # Run for a device is defined by model and dataset:
            run_settings: list[tuple[Model, Dataset, dict[str, str]]] = []
            # Create pairs of models and datasets:
            for run in runs_by_device[device_name]:
                model = next((x for x in self.models if x.name == run.model_name), None)
                dataset = next((x for x in self.datasets if x.name == run.dataset_name), None)
                run_settings.append((model, dataset, run.required_script_paths))
            # Order pairs by model name:
            run_settings = sorted(run_settings, key=lambda setting: setting[0].name)

            self.pipelines.append(RunPipeline(device_interface=interface, run_settings=run_settings))

    def execute_pipelines(self) -> None:
        """Execute all evaluation pipelines and registers their results."""
        for pipeline in tqdm(self.pipelines, desc="Pipeline"):
            self._execute_pipeline(pipeline)

    def _execute_pipeline(self, pipeline: RunPipeline) -> None:
        device = pipeline.device_interface.device
        while not pipeline.empty():
            model, dataset, metrics = pipeline.execute_run()
            run_id = self.reportInterface.register_run(device=device, model=model, dataset=dataset, metrics=metrics)
            if run_id:
                self.run_ids.append(run_id)
        
        pipeline.device_interface.shutdownApi()

    def get_report(self) -> Path | None:
        """Generate report for the experiment."""
        report = self.reportInterface.generate_report(self.run_ids)

        if report:
            report_path_dir = get_settings().REPORT_SAVE_PATH
            report_path_dir.mkdir(parents=True, exist_ok=True)

            file_name = f"report-{'-'.join([str(x) for x in self.run_ids])}.html"
            report_path = report_path_dir / file_name

            with Path(report_path).open("w+") as f:
                f.write(report)

            return report_path

        return None
