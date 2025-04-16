import json
from pathlib import Path

from .schemas import Config, Dataset, Device, Model, Run  # , DownloaderConfiguration


class ConfigParser:
    """Configuration parser - class for parsing given configuration file (--cfg argument).

    The configuration file is parsed and validated during instance initialization.
    """

    def __init__(self, path_cfg: str) -> None:
        """Create ConfigParser instance.

        :param path_cfg: JSON configuration path
        """
        # Read JSON:
        self.dictCfg = self._read_cfg(path_cfg)

        self.cfg = Config.model_validate(self._read_cfg(path_cfg))

        # TODO: distinct names reimplement (check for name conflicts within each of: models, datasets, devices)

        self._filter_wrt_runs()

    def _read_cfg(self, path: str) -> dict:
        """Load json file."""
        with Path(path).open("r") as f:
            return json.load(f)

    def _filter_wrt_runs(self) -> None:
        # Check if all runs work with valid names:
        def valid_run(r: Run) -> tuple[bool, str]:
            if r.model_name not in [x.name for x in self.cfg.models]:
                return False, r.model_name
            if r.dataset_name not in [x.name for x in self.cfg.datasets]:
                return False, r.dataset_name
            if r.device_name not in [x.name for x in self.cfg.devices]:
                return False, r.device_name
            return True, ""

        for r in self.cfg.runs:
            valid, ref = valid_run(r)
            if not valid:
                self.cfg.runs.remove(r)
                print(f"Run {vars(r)} contains invalid reference `{ref}`")

        # Remove items not mentioned in runs:
        used_model_names = {x.model_name for x in self.cfg.runs}  # Names of models used in any run
        used_dataset_names = {x.dataset_name for x in self.cfg.runs}  # Names of datasets used in any run
        used_devices_names = {x.device_name for x in self.cfg.runs}  # Names of devices used in any run

        self.cfg.models = [x for x in self.cfg.models if x.name in used_model_names]  # Filtered models
        self.cfg.datasets = [x for x in self.cfg.datasets if x.name in used_dataset_names]  # Filtered datasets
        self.cfg.devices = [x for x in self.cfg.devices if x.name in used_devices_names]  # Filtered devices

    def get_downloader_cfg(self) -> tuple[list[Model], list[Dataset]]:
        """Return configuration for Downloader component.

        Dictionary containing parsed models and datasets (models and datasets defined in `run` only).
        """
        return self.cfg.models, self.cfg.datasets

    def get_evaluator_cfg(self) -> tuple[list[Device], list[Run]]:
        """Return configuration for Model Evaluator component (parsed devices and runs)."""
        return self.cfg.devices, self.cfg.runs
