import json

from .schemas import Config, Dataset, Device, Model, Run  # , DownloaderConfiguration


class ConfigParser:
    """Configuration parser - class for parsing given configuration file (--cfg argument).
    The configuration file is parsed and validated during instance initialization.
    """

    def __init__(self, pathCfg: str):
        try:
            # Read JSON:
            self.dictCfg = self._readCfg(pathCfg)

            self.cfg = Config.model_validate(self._readCfg(pathCfg))

            # TODO: distinct names reimplement (check for name conflicts within each of: models, datasets, devices)

            self._filterWrtRuns()
        except Exception as ex:
            raise ex

    def _readCfg(self, path: str) -> dict:
        """Load json file."""
        with open(path) as f:
            return json.load(f)

    def _distinctAttributes(self, objects: list, groupName: str, attributes: list[str]):
        """Check for conflicts in given attributes within objects of same class."""
        for attr in attributes:
            if len({o.__getattribute__(attr) for o in objects}) != len(objects):
                raise Exception(f"{groupName} cannot share same {attr}")

    def _filterWrtRuns(self) -> None:
        # Check if all runs work with valid names:
        def valid_run(r: Run) -> tuple[bool, str]:
            if r.modelName not in [x.name for x in self.cfg.models]:
                return False, r.modelName
            if r.datasetName not in [x.name for x in self.cfg.datasets]:
                return False, r.datasetName
            if r.deviceName not in [x.name for x in self.cfg.devices]:
                return False, r.deviceName
            return True, ""

        for r in self.cfg.runs:
            valid, ref = valid_run(r)
            if not valid:
                self.cfg.runs.remove(r)
                print(f"Run {vars(r)} contains invalid reference `{ref}`")

        # Remove items not mentioned in runs:
        usedModelNames = {x.modelName for x in self.cfg.runs}  # Names of models used in any run
        usedDatasetNames = {x.datasetName for x in self.cfg.runs}  # Names of datasets used in any run
        usedDevicesNames = {x.deviceName for x in self.cfg.runs}  # Names of devices used in any run

        self.cfg.models = [x for x in self.cfg.models if x.name in usedModelNames]  # Filtered models
        self.cfg.datasets = [x for x in self.cfg.datasets if x.name in usedDatasetNames]  # Filtered datasets
        self.cfg.devices = [x for x in self.cfg.devices if x.name in usedDevicesNames]  # Filtered devices

    def getDownloaderCfg(self) -> tuple[list[Model], list[Dataset]]:
        """Returns configuration for Downloader component. Dictionary containing parsed models and datasets (models and datasets defined in `run` only)."""
        return self.cfg.models, self.cfg.datasets

    def getEvaluatorCfg(self) -> tuple[list[Device], list[Run]]:
        """Returns configuration for Model Evaluator component (parsed devices and runs)"""
        return self.cfg.devices, self.cfg.runs
