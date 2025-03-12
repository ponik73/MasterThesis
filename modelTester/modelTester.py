import argparse
import numpy as np

from configurationHandler.cfgParser import ConfigParser, cfgDesc
from downloader.controller import DownloaderController
from modelEvaluator.controller import EvaluatorController
from configuration import getSettings

argParser = argparse.ArgumentParser()
argParser.add_argument("--cfg", type=str, action="store", dest="path_cfg", help="Path to config")
argParser.add_argument("--cfg-help", default=False, action="store_true", help="Prints configuration file description.")

argParser.usage = f'<pyhon> modelTester.py --cfg=<PATH_CFG>'
argParser.description = "a"

# TODO: suppress the prints from the libraries; print only this program messages
# import sys
# print("aa", file=sys.__stdout__)

if __name__ == "__main__":
    SEED = 690
    np.random.seed(SEED)

    args = argParser.parse_args()

    # Print configuration file description:
    if args.cfg_help:
        print(cfgDesc)
        exit(0)
    # Required argument for application run:
    if not args.path_cfg:
        print("the following arguments are required: --cfg")
        exit(2)

    # Configuration handler - parse cfg file:
    cfgParser = ConfigParser(args.path_cfg)
    # Configuration for Downloader component:
    configurationDownloader = cfgParser.getDownloaderCfg()
    # Configuration for Model Evaluator component:
    devices, runs = cfgParser.getEvaluatorCfg()

    # Model Evaluator - initialization:
    print("#"*10 + "\n" + "DISCOVERING DEVICES\n" + "#"*10 + "\n")
    evaluator = EvaluatorController(devices, runs)

    # Download items from model hubs:
    print("#"*10 + "\n" + "RETRIEVING DATA FROM MODEL HUBS\n" + "#"*10 + "\n")
    downloader = DownloaderController(configurationDownloader)
    models, datasets = downloader.download()
    # Pass downloaded items to the Model Evaluator:
    evaluator.setModels(models)
    evaluator.setDatasets(datasets)

    print("#"*10 + "\n" + "EXECUTING RUNS\n" + "#"*10 + "\n")
    evaluator.createPipelines()
    evaluator.executePipelines()

    print("#"*10 + "\n" + "GENERATING REPORT\n" + "#"*10 + "\n")
    report_path = evaluator.get_report()
    
    if report_path:    
        print(f"Report saved at: {report_path.resolve()}")
        if getSettings().REPORT_OPEN_IN_BROWSER:
            import webbrowser
            webbrowser.open(f"file://{report_path.resolve()}")


    

