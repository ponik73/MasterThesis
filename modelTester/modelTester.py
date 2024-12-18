# python modelTester.py --cfg=ahoj
# python modelTester.py --cfg=../test/configs/first_cfg.json

import argparse
from configurationHandler.cfgParser import ConfigParser
from downloader import controller

argParser = argparse.ArgumentParser()
argParser.add_argument("--cfg", type=str, required=True, action="store", dest="path_cfg", help="Path to config")

if __name__ == "__main__":
    args = argParser.parse_args()

    cfgParser = ConfigParser(args.path_cfg)
    configurationDownloader = cfgParser.getDownloaderCfg()
    configurationModelEvaluator = cfgParser.getEvaluatorCfg()

    # downloader = 

