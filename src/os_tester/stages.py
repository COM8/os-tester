import sys
from dataclasses import dataclass
from os import path
from typing import Any, Dict, List, Optional

import cv2
import yaml  # type: ignore


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_key(mapping: Dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in stage definition.")
    return mapping[key]


def _validate_range(value: Any, name: str, min_value: Optional[float], max_value: Optional[float]) -> float:
    if not _is_number(value):
        raise ValueError(f"Expected '{name}' to be a number, got '{type(value).__name__}'.")
    value_f = float(value)
    if min_value is not None and value_f < min_value:
        raise ValueError(f"Expected '{name}' to be >= {min_value}, got {value_f}.")
    if max_value is not None and value_f > max_value:
        raise ValueError(f"Expected '{name}' to be <= {max_value}, got {value_f}.")
    return value_f


class area:
    x1Percentage: float
    x2Percentage: float
    y1Percentage: float
    y2Percentage: float

    def __init__(self, areaDict: Dict[str, Any]):
        self.x1Percentage = _validate_range(_require_key(areaDict, "x1Percentage"), "area.x1Percentage", 0.0, 1.0)
        self.x2Percentage = _validate_range(_require_key(areaDict, "x2Percentage"), "area.x2Percentage", 0.0, 1.0)
        self.y1Percentage = _validate_range(_require_key(areaDict, "y1Percentage"), "area.y1Percentage", 0.0, 1.0)
        self.y2Percentage = _validate_range(_require_key(areaDict, "y2Percentage"), "area.y2Percentage", 0.0, 1.0)
        if self.x1Percentage >= self.x2Percentage or self.y1Percentage >= self.y2Percentage:
            raise ValueError("Expected area coordinates to satisfy x1Percentage < x2Percentage and y1Percentage < y2Percentage.")


class checkFile:
    """
    A single reference file with thresholds.
    """

    filePath: str
    fileData: cv2.typing.MatLike

    ssimGeq: float
    area: Optional[area]
    nextStage: str
    actions: List[Dict[str, Any]]

    def __init__(self, fileDict: Dict[str, Any], basePath: str):
        file_path = _require_key(fileDict, "path")
        if not isinstance(file_path, str):
            raise ValueError("Expected 'path' to be a string.")
        self.filePath = path.join(basePath, file_path)
        # Check if the reference images exist and if so load them as OpenCV object
        self.fileData = self.__load(self.filePath)
        self.ssimGeq = _validate_range(_require_key(fileDict, "ssim_geq"), "ssim_geq", 0.0, 1.0)

        self.area = None
        if "area" in fileDict:
            areaDict: Dict[str, Any] = fileDict["area"]
            self.area = area(areaDict)

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
            if not isinstance(pathDict["checks"], list):
                raise ValueError("Expected 'checks' to be a list.")
            checkDict: Dict[str, Any]
            for checkDict in pathDict["checks"]:
                if not isinstance(checkDict, dict):
                    raise ValueError("Expected each entry in 'checks' to be a mapping.")
                self.checkList.append(checkFile(checkDict, basePath))

        self.actions = pathDict["actions"] if "actions" in pathDict else list()
        self.nextStage = _require_key(pathDict, "nextStage")


class stage:
    """
    A single stage with timeout
    """

    name: str
    timeoutS: float
    pathsList: List[subPath]

    def __init__(self, stageDict: Dict[str, Any], basePath: str):
        self.name = _require_key(stageDict, "stage")
        self.timeoutS = _validate_range(_require_key(stageDict, "timeout_s"), "timeout_s", 0.0, None)

        self.pathsList = list()
        paths = _require_key(stageDict, "paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("Expected 'paths' to be a non-empty list.")
        pathDict: Dict[str, Any]
        for pathDict in paths:
            if not isinstance(pathDict, dict) or "path" not in pathDict:
                raise ValueError("Expected each entry in 'paths' to contain a 'path' mapping.")
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

        if not isinstance(stagesDict["stages"], list):
            raise ValueError("Expected 'stages' to be a list.")

        self.stagesList = list()
        stageDict: Dict[str, Any]
        for stageDict in stagesDict["stages"]:
            if not isinstance(stageDict, dict):
                raise ValueError("Expected each entry in 'stages' to be a mapping.")
            self.stagesList.append(stage(stageDict, self.basePath))

    def __init__(self, basePath: str, yamlFileName: str):
        self.basePath = basePath
        self.__load_stages(yamlFileName)
