import os
from configurationHandler.configurations import supportedModelHubs

# Supported operating system - scripts for download are implemented for the operating system
supportedOS = {
    "nt" : "Win"
}
OS_NAME = os.name

class DownloaderController():
    def __init__(self, cfg: dict):
        try:
            self.cfg = cfg
            # Determine operating system (host):
            osName = os.name
            if osName not in supportedOS.keys():
                raise Exception(f'Operating system "{osName}" is not supported.')
            self.osHost = supportedOS[osName]

            # Sort models and dataset by platform:
            models = self.sortByModelHub("models")
            datasets = self.sortByModelHub("datasets")

            # Create handlers for platforms:
            self.handlerHf = HandlerHF(models["hf"], datasets["hf"], self.osHost)
            self.handlerKaggle = HandlerKaggle(models["kaggle"], datasets["kaggle"], self.osHost)
        except Exception as ex:
            raise ex
        
    def sortByModelHub(self, groupName: str):
        group = {}
        for modelHub in supportedModelHubs:
            group[modelHub] = [x for x in self.cfg[groupName] if x.platform == modelHub]
        return group
    
    def download(self):
        return "downloaded Models info", "downloaded Datasets info"
            
class DownloadHandler():
    def __init__(self, models, datasets, osHost):
        self.models = models
        self.datasets = datasets
        self.osHost = osHost

class HandlerHF(DownloadHandler):
    def __init__(self, models, datasets, osHost):
        super().__init__(models, datasets, osHost)
    
class HandlerKaggle(DownloadHandler):
    def __init__(self, models, datasets, osHost):
        super().__init__(models, datasets, osHost)
