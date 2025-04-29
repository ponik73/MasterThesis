import argparse
import sys
from colorama import init

import numpy as np
from configuration import get_settings
from configurationHandler.cfgParser import ConfigParser
from downloader.controller import DownloaderController
from modelEvaluator.controller import EvaluatorController
from utils import print_top_level, print_success

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--cfg", type=str, action="store", dest="path_cfg", help="Path to config")

arg_parser.usage = "<pyhon> modelTester.py --cfg=<PATH_CFG>"
arg_parser.description = "a"

if __name__ == "__main__":
    init(autoreset=True)
    SEED = 690
    rng = np.random.default_rng(SEED)
    rng.normal()

    args = arg_parser.parse_args()

    # Required argument for application run:
    if not args.path_cfg:
        print("the following arguments are required: --cfg")
        sys.exit(2)

    # Configuration handler - parse cfg file:
    cfg_parser = ConfigParser(args.path_cfg)
    # Configuration for Downloader component:
    models, datasets = cfg_parser.get_downloader_cfg()
    # Configuration for Model Evaluator component:
    devices, runs = cfg_parser.get_evaluator_cfg()
    # del cfg_parser

    # Model Evaluator - initialization:
    # print("#" * 10 + "\n" + "DISCOVERING DEVICES\n" + "#" * 10 + "\n")
    print_top_level("DISCOVERING DEVICES")
    evaluator = EvaluatorController(devices, runs)

    # Download items from model hubs:
    # print("#" * 10 + "\n" + "RETRIEVING DATA FROM MODEL HUBS\n" + "#" * 10 + "\n")
    print_top_level("RETRIEVING DATA FROM MODEL HUBS")
    downloader = DownloaderController(models, datasets)
    models, datasets = downloader.download()
    # Pass downloaded items to the Model Evaluator:
    evaluator.set_models(models)
    evaluator.set_datasets(datasets)

    # print("#" * 10 + "\n" + "EXECUTING RUNS\n" + "#" * 10 + "\n")
    print_top_level("EXECUTING RUNS")
    evaluator.create_pipelines()
    evaluator.execute_pipelines()

    # print("#" * 10 + "\n" + "GENERATING REPORT\n" + "#" * 10 + "\n")
    print_top_level("GENERATING REPORT")
    report_path = evaluator.get_report()

    if report_path:
        print_success(f"Report saved at: {report_path.resolve()}")
        if get_settings().REPORT_OPEN_IN_BROWSER:
            import webbrowser

            webbrowser.open(f"file://{report_path.resolve()}")
