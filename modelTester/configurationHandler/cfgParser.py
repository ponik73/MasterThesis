import os
from pathlib import Path
import json
from configurationHandler.configurations import Device, Model, Dataset, Run, supportedModelHubs #, DownloaderConfiguration

class ConfigParser():
    """Configuration parser - class for parsing given configuration file (--cfg argument).
    The configuration file is parsed and validated during instance initialization."""
    def __init__(self, pathCfg: str):
        try:
            # Read JSON:
            self.dictCfg = self._readCfg(pathCfg)

            # Parse each group:
            self.devices = self._parseGroup(
                groupName="devices",
                cfgClass=Device,
                attributes=Device.attributes,
                distinctAttributes=Device.distinctAttributes
            )
            self.models = self._parseGroup(
                groupName="models",
                cfgClass=Model,
                attributes=Model.attributes,
                distinctAttributes=Model.distinctAttributes
            )
            self.datasets = self._parseGroup(
                groupName="datasets",
                cfgClass=Dataset,
                attributes=Dataset.attributes,
                distinctAttributes=Dataset.distinctAttributes
            )
            self.runs = self._parseGroup(
                groupName="runs",
                cfgClass=Run,
                attributes=Run.attributes,
                distinctAttributes=Run.distinctAttributes
            )

            self._filterWrtRuns()
        except Exception as ex:
            raise ex

    def _readCfg(self, path: str) -> dict:
        """Load json file."""
        with open(path) as f:
            return json.load(f)
        
    def _checkGroup(self, groupname: str) -> dict:
        """Check if group (e.g. "devices") is present in configuration file and if it contains any objects."""
        if not self.dictCfg[groupname] or len(self.dictCfg[groupname]) < 1:
            raise Exception(f'Error while reading configuration file "{self.pathCfg}" - "{groupname}" not specified.')
        return self.dictCfg[groupname]
        
    def _checkAttributes(self, obj: dict, groupName:str, attributes: list[str]):
        """Check if object of given group (e.g. "devices") has all given attributes.

        Args:
            obj (dict): JSON-like object of given group
            groupName (str): name of given group (see `cfgParser.cfgDesc`)
            attributes (list[str]): list of required attributes of object of given group

        Raises:
            Exception: If object is missing required attribute(s).

        Returns:
            _type_: `True` if object contains all specified attributes.
        """
        if len(obj) != len(attributes) or set(obj) != set(attributes):
            raise Exception(f'Invalid {groupName} attributes')
        return True
    
    def _parseGroup(
            self, 
            groupName: str,
            cfgClass,
            attributes: list[str],
            distinctAttributes: list[str]) -> list:
        """Parse group (e.g. "devices") from config file - check if present and if objects have all specified attributes.

        Args:
            groupName (str): name of given group (see `cfgParser.cfgDesc`)
            cfgClass: class representing objects of given group (see `configurationHandler.configurations`)
            attributes (list[str]): list of attributes of objects of given group (see `cfgParser.cfgDesc`)
            distinctAttributes (list[str]): list of distinct attributes (no conflicts allowed) of objects of given group (see `cfgParser.cfgDesc`)

        Returns:
            list: List of parsed `cfgClass` instances.
        """
        # Load group if present in JSON.
        groupDict = self._checkGroup(groupName)

        # If each object in group has all required attributes, initialize instance of corresponding class:
        cfgObjects = []
        for obj in groupDict:
            if self._checkAttributes(obj, groupName, attributes):
                cfgObjects.append(cfgClass(**obj))

        # Check for conflicts within objects:
        self._distinctAttributes(cfgObjects, groupName, distinctAttributes)

        return cfgObjects
    
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
            if not r.modelName in [x.name for x in self.models]:
                return False, r.modelName
            if not r.datasetName in [x.name for x in self.datasets]:
                return False, r.datasetName
            if not r.deviceName in [x.name for x in self.devices]:
                return False, r.deviceName
            return True, ""
        for r in self.runs:
            valid, ref = validRun(r)
            if not valid:
                self.runs.remove(r)
                print(f'Run {vars(r)} contains invalid reference `{ref}`')

        # Remove items not mentioned in runs:
        usedModelNames = set([x.modelName for x in self.runs]) # Names of models used in any run
        usedDatasetNames = set([x.datasetName for x in self.runs]) # Names of datasets used in any run
        usedDevicesNames = set([x.deviceName for x in self.runs]) # Names of devices used in any run

        self.models = [x for x in self.models if x.name in usedModelNames] # Filtered models
        self.datasets = [x for x in self.datasets if x.name in usedDatasetNames] # Filtered datasets
        self.devices = [x for x in self.devices if x.name in usedDevicesNames] # Filtered devices

    def getDownloaderCfg(self) -> dict:
        """Returns configuration for Downloader component. Dictionary containing parsed models and datasets (models and datasets defined in `run` only)."""
        
        return {
            "models": self.models,
            "datasets": self.datasets
        }
    def getEvaluatorCfg(self):
        """Returns configuration for Model Evaluator component. Dictionary containing parsed devices, models, datasets, and runs."""
        return {
            "devices": self.devices,
            "models": self.models,
            "datasets": self.datasets,
            "runs": self.runs
        }
    
cfgDesc = """
Configuration file must contain arrays of objects specified below.

Arrays:
------------------
- devices,
- models,
- datasets,
- runs.

Object attributes:
------------------
Devices:
- name (custom device name),
- uri (device's reachable location).
Models:
- name (custom model name),
- task """ + str(Model.supportedTasks) + """,
- input_shape (list of integers representing model's input shape),
- platform """ + str(supportedModelHubs) + """,
- uri (location within platform).
Datasets:
- name (custom dataset name),
- platform """ + str(supportedModelHubs) + """,
- uri (location within platform).
Runs:
- device (name of the selected device),
- model (name of the model device),
- dataset (name of the dataset device),
"""