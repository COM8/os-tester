import json
import sys
from contextlib import suppress
from os import path, remove
from time import sleep, time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import libvirt
import libvirt_qemu
import numpy as np
from skimage import metrics as skimage_metrics

from os_tester.debug_plot import debugPlot
from os_tester.stages import stage, stages, subPath


class vm:
    """
    A wrapper around a qemu libvirt VM that handles the live time and stage execution.
    """

    conn: libvirt.virConnect
    uuid: str
    debugPlt: bool

    vmDom: Optional[libvirt.virDomain]
    debugPlotObj: debugPlot

    def __init__(
        self,
        conn: libvirt.virConnect,
        uuid: str,
        debugPlt: bool = False,
    ):
        self.conn = conn
        self.uuid = uuid
        self.debugPlt = debugPlt
        if self.debugPlt:
            self.debugPlotObj = debugPlot()

        self.vmDom = None

    def __perform_stage_actions(self, actions: List[Dict[str, Any]]) -> None:
        """
        Performs all stage actions (mouse_move, keyboard_key, reboot, ...) on the current VM.

        Args:
            actions (List[Dict[str, Any]]): A list of actions that should be performed
        """
        action: Dict[str, Any]
        for action in actions:
            if "mouse_move" in action:
                self.__send_mouse_move_action(action["mouse_move"])
            elif "mouse_click" in action:
                self.__send_mouse_click_action(action["mouse_click"])
            elif "keyboard_key" in action:
                self.__send_keyboard_key_action(action["keyboard_key"])
            elif "keyboard_text" in action:
                self.__send_keyboard_text_action(action["keyboard_text"])
            elif "sleep" in action:
                sleep(action["duration_s"])
            elif "reboot" in action:
                assert self.vmDom
                self.vmDom.reboot()
            elif "shutdown" in action:
                assert self.vmDom
                self.vmDom.shutdown()
            else:
                raise Exception(f"Invalid stage action: {action}")

    def __img_mse(
        self,
        curImg: cv2.typing.MatLike,
        refImg: cv2.typing.MatLike,
    ) -> Tuple[float, cv2.typing.MatLike]:
        """
        Calculates the mean square error between two given images.
        Both images have to have the same size.

        Args:
            curImg (cv2.typing.MatLike): The current image taken from the VM.
            refImg (cv2.typing.MatLike): The reference image we are awaiting.

        Returns:
            Tuple[float, cv2.typing.MatLike]: A tuple of the mean square error and the image diff.
        """
        # Compute the difference
        imgDif: cv2.typing.MatLike = cv2.subtract(curImg, refImg)
        err = np.sum(imgDif**2)

        # Compute Mean Squared Error
        h, w = curImg.shape[:2]
        mse = err / (float(h * w))

        mse = min(
            mse,
            10,
        )  # Values over 10 do not make sense for our case and it makes it easier to plot it
        return mse, imgDif

    def __comp_images(
        self,
        curImg: cv2.typing.MatLike,
        refImg: cv2.typing.MatLike,
    ) -> Tuple[float, float, cv2.typing.MatLike]:
        """
        Compares the provided images and calculates the mean square error and structural similarity index.
        Based on: https://www.tutorialspoint.com/how-to-compare-two-images-in-opencv-python

        Args:
            curImg (cv2.typing.MatLike): The current image taken from the VM.
            refImg (cv2.typing.MatLike): The reference image we are awaiting.

        Returns:
            Tuple[float, float, cv2.typing.MatLike]: A tuple consisting of the mean square error, structural similarity index and a image diff of both images.
        """
        # Get the dimensions of the original image
        hRef, wRef = refImg.shape[:2]

        # Resize the reference image to match the original image's dimensions
        curImgResized = cv2.resize(curImg, (wRef, hRef))

        mse: float
        difImg: cv2.typing.MatLike
        mse, difImg = self.__img_mse(curImgResized, refImg)

        # Compute SSIM
        ssimIndex: float = skimage_metrics.structural_similarity(curImgResized, refImg, channel_axis=-1)

        return (mse, ssimIndex, difImg)

    def __wait_for_stage_done(self, stageObj: stage) -> subPath:
        """
        Returns once the given stages reference image is reached.

        Args:
            stageObj (stage): The stage we want to await for.
        """
        timeoutInS = stageObj.timeoutS
        start = time()

        # Check if the reference images exist and if so load them as OpenCV object
        refImgList: List[cv2.typing.MatLike] = list()
        for subPathObj in stageObj.pathsList:
            refImgPath: str = subPathObj.checkFile
            if not path.exists(refImgPath):
                print(f"Stage ref image file '{refImgPath}' not found!")
                sys.exit(2)

            if not path.isfile(refImgPath):
                print(f"Stage ref image file '{refImgPath}' is no file!")
                sys.exit(3)

            refImgList.append(cv2.imread(refImgPath))

        while True:
            # Take a new screenshot
            curImgPath: str = f"/tmp/{self.uuid}_check.png"
            self.take_screenshot(curImgPath)
            curImg: cv2.typing.MatLike = cv2.imread(curImgPath)

            mse: float
            ssimIndex: float
            difImg: cv2.typing.MatLike
            refImg: cv2.typing.MatLike

            # Compare the screenshot with all reference images
            resultIndex: int = -1
            index: int = 0
            for subPathObj in stageObj.pathsList:
                # Compare images by calculating similarity
                refImg = refImgList[index]
                mse, ssimIndex, difImg = self.__comp_images(curImg, refImg)
                same: float = 1 if mse <= subPathObj.checkMseLeq and ssimIndex >= subPathObj.checkSsimGeq else 0

                print(f"{subPathObj.checkFile} with MSE leq {subPathObj.checkMseLeq} and SSIM geq {subPathObj.checkSsimGeq} - MSE: {mse}, SSIM: {ssimIndex}, Images Same: {same}")
                if self.debugPlt:
                    self.debugPlotObj.update_plot(refImg, curImg, difImg, mse, ssimIndex, same)

                # Break if we found a matching image
                if same >= 1:
                    resultIndex = index
                    break

                index += 1

            if resultIndex != -1:
                print(f"path number: {index}")
                return stageObj.pathsList[resultIndex]

            # if timeout is exited
            if start + timeoutInS < time():
                print(f"Timeout for stage '{stageObj.name}' reached after {timeoutInS} seconds.")
                sys.exit(5)

            sleep(0.25)

    def __run_stage(self, stageObj: stage) -> str:
        """
        1. Awaits until we reach the current stage reference image.
        2. Executes all actions defined by this stage.

        Args:
            stageObj (stage): The stage to execute/await for the image.
        Returns:
            str: with the name of the next requested Stage
        """
        start: float = time()
        print(f"Running stage '{stageObj.name}'.")

        subPathObj: subPath = self.__wait_for_stage_done(stageObj)
        self.__perform_stage_actions(subPathObj.actions)

        duration: float = time() - start
        print(f"Stage '{stageObj.name}' finished after {duration}s. Next Stage is: '{subPathObj.nextStage}'")

        return subPathObj.nextStage

    def run_stages(self, stagesObj: stages) -> None:
        """
        Executes all stages defined for the current PC and awaits every stage to finish before returning.
        If no name with requested StageName found exit with error
        """
        nextStage = stagesObj.stagesList[0]
        while True:
            nextStageName = self.__run_stage(nextStage)
            # if nextStageName is None exit program
            if nextStageName == "None":
                break
            found: bool = False
            for stageObj in stagesObj.stagesList:
                # If the expected next Stage name is found break and continue with that stage
                if stageObj.name == nextStageName:
                    nextStage = stageObj
                    found = True
                    break

            # Exit if no matching stage was found
            if not found:
                print(f"No Stage named '{nextStageName}' was found ")
                sys.exit(10)

    def try_load(self) -> bool:
        """
        Tries to lookup and load the qemu/libvirt VM via 'self.uuid' and returns the result.

        Returns:
            bool: True: The VM exists and was loaded successfully.
        """
        with suppress(libvirt.libvirtError):
            self.vmDom = self.conn.lookupByUUIDString(self.uuid)
            return self.vmDom is not None
        return False

    def destroy(self) -> None:
        """
        Tell qemu/libvirt to destroy the VM defined by 'self.uuid'.

        Raises:
            Exception: In case the VM has not been loaded before via e.g. try_load(...).
        """
        if not self.vmDom:
            raise Exception("Can not destroy vm. Use try_load or create first!")

        self.vmDom.destroy()

    def create(self, vmXml: str) -> None:
        """
        Creates a new libvirt/qemu VM based on the provided libvirt XML string.
        Ref: https://libvirt.org/formatdomain.html

        Args:
            vmXml (str): The libvirt XML string defining the VM. Ref: https://libvirt.org/formatdomain.html

        Raises:
            Exception: In case the VM with 'self.uuid' already exists.
        """
        with suppress(libvirt.libvirtError):
            if self.conn.lookupByUUIDString(self.uuid):
                raise Exception(
                    f"Can not create vm with UUID '{self.uuid}'. VM already exists. Destroy first!",
                )

        self.vmDom = self.conn.createXML(vmXml, 0)

    def take_screenshot(self, targetPath: str) -> None:
        """
        Takes a screenshoot of the current VM output and stores it as a file.

        Args:
            targetPath (str): Where to store the screenshoot at.
        """
        stream: libvirt.virStream = self.conn.newStream()

        assert self.vmDom
        _ = self.vmDom.screenshot(stream, 0)

        with open(targetPath, "wb") as f:
            streamBytes = stream.recv(262120)
            while streamBytes != b"":
                f.write(streamBytes)
                streamBytes = stream.recv(262120)
        stream.finish()

    def __get_screen_size(self) -> Tuple[int, int]:
        """
        Helper function returning the VM screen size by taking a screenshoot and using this image than as width and height.

        Returns:
            Tuple[int, int]: width and height
        """
        filePath: str = f"/tmp/{self.uuid}_screen_size.png"
        self.take_screenshot(filePath)

        img: cv2.typing.MatLike = cv2.imread(filePath)

        # Delete screen shoot again since we do not need it any more
        remove(filePath)

        h, w = img.shape[:2]
        return (w, h)

    def __send_action(self, cmdDict: Dict[str, Any]) -> Optional[Any]:
        """
        Sends a qemu monitor command to the VM.
        Ref: https://en.wikibooks.org/wiki/QEMU/Monitor

        Args:
            cmdDict (Dict[str, Any]): A dict defining the qemu monitor command.

        Returns:
            Optional[Any]: The qemu execution result.
        """
        cmd: str = json.dumps(cmdDict)
        try:
            response: Any = libvirt_qemu.qemuMonitorCommand(self.vmDom, cmd, 0)
            print(f"Action response: {response}")
            return response
        except libvirt.libvirtError as e:
            print(f"Failed to send action event: {e}")
        return None

    def __send_keyboard_text_action(self, keyboardText: Dict[str, Any]) -> None:
        """
        Sends a row of key press events via the qemu monitor.

        Args:
            keyboardText (Dict[str, Any]): The dict defining the text to send and how.
        """
        for c in keyboardText["value"]:
            cmdDictDown: Dict[str, Any] = {
                "execute": "input-send-event",
                "arguments": {
                    "events": [
                        {
                            "type": "key",
                            "data": {
                                "down": True,
                                "key": {"type": "qcode", "data": c},
                            },
                        },
                    ],
                },
            }
            self.__send_action(cmdDictDown)
            sleep(keyboardText["duration_s"])

            cmdDictUp: Dict[str, Any] = {
                "execute": "input-send-event",
                "arguments": {
                    "events": [
                        {
                            "type": "key",
                            "data": {
                                "down": False,
                                "key": {"type": "qcode", "data": c},
                            },
                        },
                    ],
                },
            }
            self.__send_action(cmdDictUp)
            sleep(keyboardText["duration_s"])

    def __send_keyboard_key_action(self, keyboardKey: Dict[str, Any]) -> None:
        """
        Performs a keyboard key press action via the qemu monitor.

        Args:
            keyboardKey (Dict[str, Any]): The dict defining the keyboard key to send and how.
        """
        cmdDictDown: Dict[str, Any] = {
            "execute": "input-send-event",
            "arguments": {
                "events": [
                    {
                        "type": "key",
                        "data": {
                            "down": True,
                            "key": {"type": "qcode", "data": keyboardKey["value"]},
                        },
                    },
                ],
            },
        }
        self.__send_action(cmdDictDown)
        sleep(keyboardKey["duration_s"])

        cmdDictUp: Dict[str, Any] = {
            "execute": "input-send-event",
            "arguments": {
                "events": [
                    {
                        "type": "key",
                        "data": {
                            "down": False,
                            "key": {"type": "qcode", "data": keyboardKey["value"]},
                        },
                    },
                ],
            },
        }
        self.__send_action(cmdDictUp)
        sleep(keyboardKey["duration_s"])

    def __send_mouse_move_action(self, mouseMove: Dict[str, Any]) -> None:
        """
        Performs a mouse move action via the qemu monitor.

        Args:
            mouseMove (Dict[str, Any]): The dict defining the mouse move action.
        """
        w: int
        h: int
        w, h = self.__get_screen_size()

        cmdDict: Dict[str, Any] = {
            "execute": "input-send-event",
            "arguments": {
                "events": [
                    {
                        "type": "abs",
                        "data": {
                            "axis": "x",
                            "value": 0,
                        },
                    },
                    {
                        "type": "abs",
                        "data": {"axis": "y", "value": 0},
                    },
                    {
                        "type": "rel",
                        "data": {
                            "axis": "x",
                            "value": int(w * mouseMove["x_rel"]),
                        },
                    },
                    {
                        "type": "rel",
                        "data": {"axis": "y", "value": int(h * mouseMove["y_rel"])},
                    },
                ],
            },
        }
        self.__send_action(cmdDict)
        sleep(mouseMove["duration_s"])

    def __send_mouse_click_action(self, mouseClick: Dict[str, Any]) -> None:
        """
        Performs a mouse click action via the qemu monitor.

        Args:
            mouseMove (Dict[str, Any]): The dict defining the mouse click action.
        """
        cmdDictDown: Dict[str, Any] = {
            "execute": "input-send-event",
            "arguments": {
                "events": [
                    {
                        "type": "btn",
                        "data": {"down": True, "button": mouseClick["value"]},
                    },
                ],
            },
        }
        self.__send_action(cmdDictDown)
        sleep(mouseClick["duration_s"])

        cmdDictUp: Dict[str, Any] = {
            "execute": "input-send-event",
            "arguments": {
                "events": [
                    {
                        "type": "btn",
                        "data": {"down": False, "button": mouseClick["value"]},
                    },
                ],
            },
        }
        self.__send_action(cmdDictUp)
        sleep(mouseClick["duration_s"])
