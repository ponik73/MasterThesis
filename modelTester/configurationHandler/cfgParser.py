import os
from pathlib import Path
import json
from typing import Tuple, List
from configurationHandler.configurations import Device, Model, Dataset, Run, SupportedModelHubs, Config #, DownloaderConfiguration

class ConfigParser():
    """Configuration parser - class for parsing given configuration file (--cfg argument).
    The configuration file is parsed and validated during instance initialization."""
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
        """Checks for conflicts in given attributes within objects of same class.

        Args:
            objects (list): list of objects of given class (e.g. `configuration.Model`).
            groupName (str): name of given group (see `cfgParser.cfgDesc`).
            attributes (list[str]): names of attributes that can't be same.

        Raises:
            Exception: If there are any conflicts within given attributes.
        """
        for attr in attributes:
            if len(set([o.__getattribute__(attr) for o in objects])) != len(objects):
                raise Exception(f'{groupName} cannot share same {attr}')
            
    def _filterWrtRuns(self):
        # Check if all runs work with valid names:
        def validRun(r: Run) -> tuple[bool, str]:
            if not r.modelName in [x.name for x in self.cfg.models]:
                return False, r.modelName
            if not r.datasetName in [x.name for x in self.cfg.datasets]:
                return False, r.datasetName
            if not r.deviceName in [x.name for x in self.cfg.devices]:
                return False, r.deviceName
            return True, ""
        for r in self.cfg.runs:
            valid, ref = validRun(r)
            if not valid:
                self.cfg.runs.remove(r)
                print(f'Run {vars(r)} contains invalid reference `{ref}`')

        # Remove items not mentioned in runs:
        usedModelNames = set([x.modelName for x in self.cfg.runs]) # Names of models used in any run
        usedDatasetNames = set([x.datasetName for x in self.cfg.runs]) # Names of datasets used in any run
        usedDevicesNames = set([x.deviceName for x in self.cfg.runs]) # Names of devices used in any run

        self.cfg.models = [x for x in self.cfg.models if x.name in usedModelNames] # Filtered models
        self.cfg.datasets = [x for x in self.cfg.datasets if x.name in usedDatasetNames] # Filtered datasets
        self.cfg.devices = [x for x in self.cfg.devices if x.name in usedDevicesNames] # Filtered devices

    def getDownloaderCfg(self) -> Tuple[List[Model], List[Dataset]]:
        """Returns configuration for Downloader component. Dictionary containing parsed models and datasets (models and datasets defined in `run` only)."""
        return self.cfg.models, self.cfg.datasets
    
    def getEvaluatorCfg(self) -> Tuple[List[Device], List[Run]]:
        """Returns configuration for Model Evaluator component (parsed devices and runs)"""
        return self.cfg.devices, self.cfg.runs
    
# TODO: rework
cfgDesc = "" 
# cfgDesc = """
# Configuration file must contain arrays of objects specified below.

# Arrays:
# ------------------
# - devices,
# - models,
# - datasets,
# - runs.

# Object attributes:
# ------------------
# Devices:
# - name (custom device name),
# - uri (device's reachable location).
# Models:
# - name (custom model name),
# - task """ + str(Model.SupportedTasks.__members__.items()) + """,
# - input_shape (list of integers representing model's input shape),
# - platform """ + str(supportedModelHubs) + """,
# - uri (location within platform).
# Datasets:
# - name (custom dataset name),
# - platform """ + str(supportedModelHubs) + """,
# - uri (location within platform).
# Runs:
# - device (name of the selected device),
# - model (name of the model device),
# - dataset (name of the dataset device),
# """