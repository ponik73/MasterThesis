# python modelTester.py --cfg=ahoj
# python modelTester.py --cfg=../test/configs/first_cfg.json

import argparse
from configurationHandler.cfgParser import ConfigParser, cfgDesc
# from downloader.controller import DownloaderController

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
    configurationModelEvaluator = cfgParser.getEvaluatorCfg()

    # # Initialize Downloader component:
    # downloader = DownloaderController(configurationDownloader)
    # m, d = downloader.download()
    # print(m)
    # print(d)

