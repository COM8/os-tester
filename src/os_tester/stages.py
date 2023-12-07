import yaml
from os import path
from typing import Dict, Any, List


class stage:
    name: str
    timeoutS: float
    checkFile: str
    checkMseLeq: str
    checkSsimGeq: str
    actions: List[Dict[str, Any]]

    def __init__(self, stage: Dict[str, Any], basePath: str):
        self.name = stage["stage"]
        self.timeoutS = stage["timeout_s"]
        self.checkFile = path.join(basePath, stage["check"]["file"])
        self.checkMseLeq = stage["check"]["mse_leq"]
        self.checkSsimGeq = stage["check"]["ssim_geq"]
        self.actions = stage["actions"] if "actions" in stage else list()


class stages:
    basePath: str
    stagesList: List[stage]

    def __load_stages(self) -> None:
        ymlFile: str = path.join(self.basePath, "stages.yml")
        print(f"Loading stages from: {ymlFile}")

        if not path.exists(ymlFile):
            print(f"Stage config at '{ymlFile}' not found!")
            exit(2)

        if not path.isfile(ymlFile):
            print(f"Stage config at '{ymlFile}' is no file!")
            exit(3)

        stagesDict: Dict[str, Any]
        with open(ymlFile, "r") as file:
            stagesDict = yaml.safe_load(file)

        self.stagesList = list()
        stageDict: Dict[str, Any]
        for stageDict in stagesDict["stages"]:
            self.stagesList.append(stage(stageDict, self.basePath))

    def __init__(self, basePath: str):
        self.basePath = basePath
        self.__load_stages()
