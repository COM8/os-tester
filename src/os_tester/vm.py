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
from os_tester.stages import area, stage, stages, subPath


class vm:
    """
    A wrapper around a qemu libvirt VM that handles the live time and stage execution.
    """

    conn: libvirt.virConnect
    uuid: str
    debugPlt: bool

    vmDom: Optional[libvirt.virDomain]
    debugPlotObj: debugPlot
    matchedImageIndex: int

    def __init__(self, conn: libvirt.virConnect, uuid: str, debugPlt: bool = False):
        self.conn = conn
        self.uuid = uuid
        self.debugPlt = debugPlt
        if self.debugPlt:
            self.debugPlotObj = debugPlot()

        self.vmDom = None
        self.matchedImageIndex = 0

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
                print(f"Sleeping for {action["duration_s"]} seconds...")
                sleep(action["duration_s"])
            elif "reboot" in action:
                print("Rebooting VM...")
                assert self.vmDom
                self.vmDom.reboot()
            elif "shutdown" in action:
                print("Shutting Down VM...")
                assert self.vmDom
                self.vmDom.shutdown()
            else:
                raise Exception(f"Invalid stage action: {action}")

    def __img_diff(
        self,
        curImg: cv2.typing.MatLike,
        refImg: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        """
        Calculates the image difference between two given images.
        Both images have to have the same size.

        Args:
            curImg (cv2.typing.MatLike): The current image taken from the VM.
            refImg (cv2.typing.MatLike): The reference image we are awaiting.

        Returns:
            Tcv2.typing.MatLike: The image diff.
        """
        # Use absdiff on float to avoid uint8 saturation masking differences.
        imgDiff: cv2.typing.MatLike = cv2.absdiff(
            curImg.astype(np.float32),
            refImg.astype(np.float32),
        )
        return imgDiff

    def __comp_images(
        self,
        curImg: cv2.typing.MatLike,
        refImg: cv2.typing.MatLike,
        imageArea: Optional[area] = None,
    ) -> Tuple[float, cv2.typing.MatLike]:
        """
        Compares the provided images and calculates the structural similarity index.
        Based on: https://scikit-image.org/docs/0.25.x/auto_examples/transform/plot_ssim.html

        Args:
            curImg (cv2.typing.MatLike): The current image taken from the VM.
            refImg (cv2.typing.MatLike): The reference image we are awaiting.
            area (Optional[Area]): Optional sub-rectangle (normalized) used for comparison.

        Returns:
            Tuple[float, cv2.typing.MatLike]: A tuple consisting of the structural similarity index and a image diff of both images.
        """
        # Get the dimensions of the original image
        hRef, wRef = refImg.shape[:2]

        # Get the dimensions of the current image
        hCur, wCur = curImg.shape[:2]

        # Resize the reference image to match the original image's dimensions
        if (hRef != hCur) or (wRef != wCur):
            curImgResized = cv2.resize(curImg, (wRef, hRef))
        else:
            curImgResized = curImg

        # If a sub-area has been defined, cut the image accordingly
        if imageArea is not None:
            x1 = int(np.floor(imageArea.x1Percentage * wRef))
            x2 = int(np.ceil(imageArea.x2Percentage * wRef))
            y1 = int(np.floor(imageArea.y1Percentage * hRef))
            y2 = int(np.ceil(imageArea.y2Percentage * hRef))
            refImg = refImg[y1:y2, x1:x2]
            curImgResized = curImgResized[y1:y2, x1:x2]

        diffImg: cv2.typing.MatLike = self.__img_diff(refImg, curImgResized)

        ssimIndex: float = skimage_metrics.structural_similarity(refImg, curImgResized, channel_axis=-1)
        ssimIndex = min(1.0, max(0.0, ssimIndex))

        # ssimIndex = np.clip(a=ssimIndex, a_min=0.0, a_max=1.0)
        return (ssimIndex, diffImg)

    def __draw_area_outline(self, img: cv2.typing.MatLike, imageArea: area) -> cv2.typing.MatLike:
        """
        Draws a red outline around the selected area on a copy of the provided image.
        """
        h, w = img.shape[:2]
        x1 = int(np.floor(imageArea.x1Percentage * w))
        x2 = int(np.ceil(imageArea.x2Percentage * w))
        y1 = int(np.floor(imageArea.y1Percentage * h))
        y2 = int(np.ceil(imageArea.y2Percentage * h))

        x1 = max(0, min(w - 1, x1))
        x2 = max(1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(1, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return img

        width = x2 - x1
        height = y2 - y1

        # Reduce outline thickness on edges that touch the image borders.
        left_thickness = min(1 if x1 == 0 else 2, width)
        right_thickness = min(1 if x2 == w else 2, width)
        top_thickness = min(1 if y1 == 0 else 2, height)
        bottom_thickness = min(1 if y2 == h else 2, height)

        outlined = img.copy()
        red = (0, 0, 255)

        outlined[y1:y2, x1 : x1 + left_thickness] = red
        outlined[y1:y2, x2 - right_thickness : x2] = red
        outlined[y1 : y1 + top_thickness, x1:x2] = red
        outlined[y2 - bottom_thickness : y2, x1:x2] = red

        return outlined

    def __save_matched_image(self, curImg: cv2.typing.MatLike, imageArea: Optional[area]) -> None:
        """
        Stores the matched image under '/tmp/matched_{self.uuid}_{self.matchedImageIndex}.png'.
        Adds a red outline if an area-based comparison was used.
        Increments 'self.matchedImageIndex' by one.

        Args:
            curImg (cv2.typing.MatLike): The source image to save.
            imageArea (Optional[area]): Optional comparison area to outline.
        """
        if imageArea is not None:
            curImg = self.__draw_area_outline(curImg, imageArea)

        cv2.imwrite(f"/tmp/matched_{self.uuid}_{self.matchedImageIndex}.png", curImg)
        self.matchedImageIndex += 1

    def __wait_for_stage_done(self, stageObj: stage) -> subPath:
        """
        Returns once the given stages reference image is reached.

        Args:
            stageObj (stage): The stage we want to await for.
        """
        timeoutInS = stageObj.timeoutS
        start = time()

        while True:
            loop_start = time()
            # Take a new screenshot
            curImgPath: str = f"/tmp/check_{self.uuid}.png"
            self.take_screenshot(curImgPath)
            curImgOpt: cv2.typing.MatLike | None = cv2.imread(curImgPath)
            if curImgOpt is None:
                print("Failed to convert current image to CV2 object")
                sys.exit(6)
            curImg: cv2.typing.MatLike = curImgOpt

            ssimIndex: float
            difImg: cv2.typing.MatLike

            pathIndex: int = 1

            # Compare the screenshot with all reference images
            for subPathObj in stageObj.pathsList:
                # If there are no checks. We consider is asd a successful check
                if not subPathObj.checkList:
                    return subPathObj

                print(f"Checking path {pathIndex}...")
                for check in subPathObj.checkList:
                    # Compare images by calculating similarity
                    ssimIndex, difImg = self.__comp_images(curImg, check.fileData, check.area)
                    same: float = 1 if ssimIndex >= check.ssimGeq else 0

                    if self.debugPlt:
                        self.debugPlotObj.update_plot(check.fileData, curImg, difImg, ssimIndex, same)

                    # Break if we found a matching image
                    if same >= 1:
                        print(f"\t✅ [{path.basename(check.filePath)}]: SSIM expected geq {check.ssimGeq} - SSIM actual: {ssimIndex}, Images same: {same}")
                        self.__save_matched_image(curImg, check.area)
                        return subPathObj
                    print(f"\t❌ [{path.basename(check.filePath)}]: SSIM expected geq {check.ssimGeq} - SSIM actual: {ssimIndex}, Images same: {same}")

                pathIndex += 1

            # if timeout is exited
            if start + timeoutInS < time():
                print(f"⌛ Timeout for stage '{stageObj.name}' reached after {timeoutInS} seconds.")
                sys.exit(5)

            # Avoid checking more frequently than every 0.5 seconds, accounting for processing time.
            elapsed = time() - loop_start
            sleep(max(0.0, 0.5 - elapsed))

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

        img: cv2.typing.MatLike | None = cv2.imread(filePath)

        if not img:
            return (0, 0)

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
