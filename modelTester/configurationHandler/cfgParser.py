import os
from pathlib import Path
import json
from configurationHandler.configurations import DownloaderConfiguration, Device, Model, Dataset, Run

class ConfigParser():
    
    def __init__(self, pathCfg: str):
        self.pathCfg = pathCfg
        self.dictCfg = self._readCfg()

        try:
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
        except Exception as ex:
            raise ex

    def _readCfg(self) -> dict:
        with open(self.pathCfg) as f:
            return json.load(f)
        
    def _checkGroup(self, groupname: str) -> dict:
        if not self.dictCfg[groupname] or len(self.dictCfg[groupname]) < 1:
            raise Exception(f'Error while reading configuration file "{self.pathCfg}" - "{groupname}" not specified.')
        return self.dictCfg[groupname]
        
    def _checkAttributes(self, obj: dict, objName:str, attributes: list[str]):
        if len(obj) != len(attributes) or set(obj) != set(attributes):
            raise Exception(f'Invalid {objName} attributes')
        return True
    
    def _parseGroup(
            self, 
            groupName: str,
            cfgClass, #TODO: class type
            attributes: list[str],
            distinctAttributes: list[str],
            ):
        groupDict = self._checkGroup(groupName)
        cfgObjects = []
        for obj in groupDict:
            if self._checkAttributes(obj, groupName, attributes):
                cfgObjects.append(cfgClass(**obj))
        self._distinctAttributes(cfgObjects, groupName, distinctAttributes)
        return cfgObjects
    
    def _distinctAttributes(self, objects: list, groupName: str, attributes: list[str]):
        for attr in attributes:
            if len(set([o.__getattribute__(attr) for o in objects])) != len(objects):
                raise Exception(f'{groupName} cannot share same {attr}')

    def getDownloaderCfg(self):
        return DownloaderConfiguration(self.models, self.datasets)
    def getEvaluatorCfg(self):
        return {
            "devices": self.devices,
            "models": self.models,
            "datasets": self.datasets,
            "runs": self.runs
        }