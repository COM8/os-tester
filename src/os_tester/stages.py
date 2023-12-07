import sys
from os import path
from typing import Any, Dict, List

import yaml  # type: ignore


class stage:
    """
    A single stage with reference image, thresholds and actions to perform once the threshold is reached.
    """

    name: str
    timeoutS: float
    checkFile: str
    checkMseLeq: float
    checkSsimGeq: float
    actions: List[Dict[str, Any]]

    def __init__(self, stageDict: Dict[str, Any], basePath: str):
        self.name = stageDict["stage"]
        self.timeoutS = stageDict["timeout_s"]
        self.checkFile = path.join(basePath, stageDict["check"]["file"])
        self.checkMseLeq = stageDict["check"]["mse_leq"]
        self.checkSsimGeq = stageDict["check"]["ssim_geq"]
        self.actions = stageDict["actions"] if "actions" in stageDict else list()


class stages:
    """
    A list of stages that are used to automate the VM process.
    """

    basePath: str
    stagesList: List[stage]

    def __load_stages(self) -> None:
        """
        Loads the stage definition from 'self.basePath' and stores the result inside 'self.stagesList'.
        """
        ymlFile: str = path.join(self.basePath, "stages.yml")
        print(f"Loading stages from: {ymlFile}")

        if not path.exists(ymlFile):
            print(f"Stage config at '{ymlFile}' not found!")
            sys.exit(2)

        if not path.isfile(ymlFile):
            print(f"Stage config at '{ymlFile}' is no file!")
            sys.exit(3)

        stagesDict: Dict[str, Any]
        with open(ymlFile, "r", encoding="utf-8") as file:
            stagesDict = yaml.safe_load(file)

        self.stagesList = list()
        stageDict: Dict[str, Any]
        for stageDict in stagesDict["stages"]:
            self.stagesList.append(stage(stageDict, self.basePath))

    def __init__(self, basePath: str):
        self.basePath = basePath
        self.__load_stages()
