import sys
from os import path
from typing import Any, Dict, List

import yaml  # type: ignore


class subPath:
    """
    A single path with reference image, thresholds and actions to perform once the threshold is reached.
    """

    checkFile: str

    checkMseLeq: float
    checkSsimGeq: float
    nextStage: str
    actions: List[Dict[str, Any]]

    def __init__(self, pathDict: Dict[str, Any], basePath: str):
        self.checkFile = path.join(basePath, pathDict["check"]["file"])
        self.checkMseLeq = pathDict["check"]["mse_leq"]
        self.checkSsimGeq = pathDict["check"]["ssim_geq"]
        self.actions = pathDict["actions"] if "actions" in pathDict else list()
        self.nextStage = pathDict["nextStage"]


class stage:
    """
    A single stage with timeout
    """

    name: str
    timeoutS: float
    pathsList: List[subPath]

    def __init__(self, stageDict: Dict[str, Any], basePath: str):
        self.name = stageDict["stage"]
        self.timeoutS = stageDict["timeout_s"]

        self.pathsList = list()
        pathDict: Dict[str, Any]
        for pathDict in stageDict["paths"]:
            self.pathsList.append(subPath(pathDict["path"], basePath))


class stages:
    """
    A list of stages that are used to automate the VM process.
    """

    basePath: str
    stagesList: List[stage]

    def __load_stages(self, yamlFileName: str) -> None:
        """
        Loads the stage definition from 'self.basePath' and stores the result inside 'self.stagesList'.
        """
        ymlFilePath: str = path.join(self.basePath, yamlFileName + ".yml")
        print(f"Loading stages from: {ymlFilePath}")

        if not path.exists(ymlFilePath):
            print(f"Stage config at '{ymlFilePath}' not found!")
            sys.exit(2)

        if not path.isfile(ymlFilePath):
            print(f"Stage config at '{ymlFilePath}' is no file!")
            sys.exit(3)

        stagesDict: Dict[str, Any]
        with open(ymlFilePath, "r", encoding="utf-8") as file:
            stagesDict = yaml.safe_load(file)

        self.stagesList = list()
        stageDict: Dict[str, Any]
        for stageDict in stagesDict["stages"]:
            self.stagesList.append(stage(stageDict, self.basePath))

    def __init__(self, basePath: str, yamlFileName: str):
        self.basePath = basePath
        self.__load_stages(yamlFileName)
