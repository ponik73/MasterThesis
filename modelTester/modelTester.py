# python modelTester.py --cfg=ahoj
# python modelTester.py --cfg=../test/configs/first_cfg.json
import os
import argparse
from configurationHandler.cfgParser import ConfigParser, cfgDesc
from downloader.controller import DownloaderController
from modelEvaluator.controller import EvaluatorController

#TODO: Maybe do this as server - request "test" with cfg file; request "report"

argParser = argparse.ArgumentParser()
argParser.add_argument("--cfg", type=str, action="store", dest="path_cfg", help="Path to config")
argParser.add_argument("--cfg-help", default=False, action="store_true", help="Prints configuration file description.")

argParser.usage = f'<pyhon> modelTester.py --cfg=<PATH_CFG>'
argParser.description = "a"

if __name__ == "__main__":
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

    # TODO: check devices first, then download models

    # Initialize Downloader component:
    downloader = DownloaderController(configurationDownloader)
    print("#"*10 + "\n" + "RETRIEVING DATA FROM MODEL HUBS\n" + "#"*10 + "\n")
    models, datasets = downloader.download()

    # Model Evaluator - initialization:
    print("#"*10 + "\n" + "DISCOVERING DEVICES\n" + "#"*10 + "\n")
    evaluator = EvaluatorController(
        devices,
        runs,
        models,
        datasets
        )
    print("#"*10 + "\n" + "EXECUTING RUNS\n" + "#"*10 + "\n")
    # Model Evaluator - execute all runs:
    while not evaluator.finished():
        evaluator.executeRun()

