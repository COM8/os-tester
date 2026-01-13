import sys
from os import path
from typing import Any, Dict, List

import cv2
import yaml  # type: ignore


class checkFile:
    """
    A single reference file with thresholds.
    """

    filePath: str
    fileData: cv2.typing.MatLike

    mseLeq: float
    ssimGeq: float
    nextStage: str
    actions: List[Dict[str, Any]]

    def __init__(self, fileDict: Dict[str, Any], basePath: str):
        self.filePath = path.join(basePath, fileDict["path"])
        # Check if the reference images exist and if so load them as OpenCV object
        self.fileData = self.__load(self.filePath)
        self.mseLeq = fileDict["mse_leq"]
        self.ssimGeq = fileDict["ssim_geq"]

    def __load(self, filePath: str) -> cv2.typing.MatLike:
        """
        Check if the reference image exist and if so load/return it as OpenCV object.
        """

        if not path.exists(filePath):
            print(f"Stage ref image file '{filePath}' not found!")
            sys.exit(2)

        if not path.isfile(filePath):
            print(f"Stage ref image file '{filePath}' is no file!")
            sys.exit(3)

        data: cv2.typing.MatLike | None = cv2.imread(filePath)
        if data is None:
            print(f"Failed to load CV2 data from '{filePath}'!")
            sys.exit(4)
        return data


class subPath:
    """
    A single path with optionally multiple ref images, thresholds and actions to perform once the threshold for one file (image) is reached.
    """

    checkList: List[checkFile]

    nextStage: str
    actions: List[Dict[str, Any]]

    def __init__(self, pathDict: Dict[str, Any], basePath: str):
        # Removed in 1.1.0
        if "check" in pathDict:
            raise Exception("The keyword 'check' has been replaced with the 'checks' keyword.")

        self.checkList = list()
        if "checks" in pathDict:
            checkDict: Dict[str, Any]
            for checkDict in pathDict["checks"]:
                self.checkList.append(checkFile(checkDict, basePath))

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
