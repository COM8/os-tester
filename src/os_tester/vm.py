import libvirt
import libvirt_qemu
from typing import Optional, Dict, Any, Tuple
import json
from time import sleep
import cv2
from os import path, remove
from os_tester.stages import stages, stage
from time import time

from os_tester.debugPlot import debugPlot
import numpy as np
from skimage.metrics import structural_similarity as ssimFunc


class vm:
    conn: libvirt.virConnect
    uuid: str
    debugPlt: bool

    vmDom: Optional[libvirt.virDomain]
    stagesObj: stages
    debugPlotObj: debugPlot

    def __init__(
        self,
        conn: libvirt.virConnect,
        uuid: str,
        stagesObj: stages,
        debugPlt: bool = False,
    ):
        self.conn = conn
        self.uuid = uuid
        self.stagesObj = stagesObj
        self.debugPlt = debugPlt
        self.debugPlotObj = debugPlot()

        self.vmDom = None

    def perform_stage_actions(self, stageObj: stage) -> None:
        for action in stageObj.actions:
            if "mouse_move" in action:
                self.send_mouse_move_action(action["mouse_move"])
            elif "mouse_click" in action:
                self.send_mouse_click_action(action["mouse_click"])
            elif "keyboard_key" in action:
                self.send_keyboard_key_action(action["keyboard_key"])
            elif "keyboard_text" in action:
                self.send_keyboard_text_action(action["keyboard_text"])
            elif "reboot" in action:
                self.vmDom.reboot()
            else:
                raise Exception(f"Invalid stage action: {action}")

    def __img_mse(
        self, curImg: cv2.typing.MatLike, refImg: cv2.typing.MatLike
    ) -> Tuple[float, cv2.typing.MatLike]:
        # Compute the difference
        imgDif: cv2.typing.MatLike = cv2.subtract(curImg, refImg)
        err = np.sum(imgDif**2)

        # Compute Mean Squared Error
        h, w = curImg.shape[:2]
        mse = err / (float(h * w))

        mse = min(
            mse, 10
        )  # Values over 10 do not make sense for our case and it makes it easier to plot it
        return mse, imgDif

    # https://www.tutorialspoint.com/how-to-compare-two-images-in-opencv-python
    def comp_images(
        self, curImg: cv2.typing.MatLike, refImg: cv2.typing.MatLike
    ) -> Tuple[float, float, cv2.typing.MatLike]:
        # Get the dimensions of the original image
        hRef, wRef = refImg.shape[:2]

        # Resize the reference image to match the original image's dimensions
        curImgResized = cv2.resize(curImg, (wRef, hRef))

        mse: float
        difImg: cv2.typing.MatLike
        mse, difImg = self.__img_mse(curImgResized, refImg)

        # Compute SSIM
        ssimIndex: float = ssimFunc(curImgResized, refImg, channel_axis=-1)

        return (mse, ssimIndex, difImg)

    def wait_for_stage_done(self, stageObj: stage) -> None:
        refImgPath: str = stageObj.checkFile
        if not path.exists(refImgPath):
            print(f"Stage ref image file '{refImgPath}' not found!")
            exit(2)

        if not path.isfile(refImgPath):
            print(f"Stage ref image file '{refImgPath}' is no file!")
            exit(3)

        refImg: cv2.typing.MatLike = cv2.imread(refImgPath)

        while True:
            curImgPath: str = f"/tmp/{self.uuid}_check.png"
            self.take_screenshot(curImgPath)
            print("Screenshoot taken.")
            curImg: cv2.typing.MatLike = cv2.imread(curImgPath)

            mse: float
            ssimIndex: float
            difImg: cv2.typing.MatLike
            mse, ssimIndex, difImg = self.comp_images(curImg, refImg)

            same: float = (
                1
                if mse < stageObj.checkMseLeq and ssimIndex > stageObj.checkSsimGeq
                else 0
            )

            print(f"MSE: {mse}, SSIM: {ssimIndex}, Images Same: {same}")
            self.debugPlotObj.update_plot(refImg, curImg, difImg, mse, ssimIndex, same)

            # Break if it's the same image
            if same >= 1:
                break
            sleep(1)

    def run_stage(self, stageObj: stage) -> None:
        start: float = time()
        print(f"Running stage '{stageObj.name}'.")

        self.wait_for_stage_done(stageObj)
        self.perform_stage_actions(stageObj)

        duration: float = time() - start
        print(f"Stage '{stageObj.name}' finished after {duration}s.")

    def run_stages(self) -> None:
        stageObj: stage
        for stageObj in self.stagesObj.stagesList:
            self.run_stage(stageObj)

    def try_load(self) -> bool:
        try:
            self.vmDom = self.conn.lookupByUUIDString(self.uuid)
            if self.vmDom:
                return True
        except libvirt.libvirtError as e:
            pass
        return False

    def destroy(self) -> None:
        if not self.vmDom:
            raise Exception("Can not destroy vm. Use try_load or create first!")

        self.vmDom.destroy()

    def create(self, vmXml: str) -> None:
        try:
            if self.conn.lookupByUUIDString(self.uuid):
                raise Exception(
                    f"Can not create vm with UUID '{self.uuid}'. VM already exists. Destroy first!"
                )
        except libvirt.libvirtError as e:
            pass

        self.vmDom = self.conn.createXML(vmXml, 0)

    def take_screenshot(self, path: str) -> None:
        stream: libvirt.virStream = self.conn.newStream()
        imgType: Any = self.vmDom.screenshot(stream, 0)

        f = open(path, "wb")
        streamBytes = stream.recv(262120)
        while streamBytes != b"":
            f.write(streamBytes)
            streamBytes = stream.recv(262120)
            f.close()

        print(f"Screenshot saved as type '{imgType}' under '{path}'.")
        stream.finish()

    def get_screen_size(self) -> Tuple[int, int]:
        filePath: str = f"/tmp/{self.uuid}_screen_size.png"
        imgPath: str = self.take_screenshot(filePath)

        img: cv2.typing.MatLike = cv2.imread(imgPath)

        # Delete screen shoot again since we do not need it any more
        remove(filePath)

        h, w = img.shape[:2]
        return (w, h)

    def send_action(self, cmdDict: Dict[str, Any]) -> Optional[Any]:
        cmd: str = json.dumps(cmdDict)
        try:
            response: Any = libvirt_qemu.qemuMonitorCommand(self.vmDom, cmd, 0)
            print(f"Action response: {response}")
            return response
        except libvirt.libvirtError as e:
            print(f"Failed to send action event: {e}")
        return None

    def send_keyboard_text_action(self, keyboardText: Dict[str, Any]) -> None:
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
                        }
                    ]
                },
            }
            self.send_action(cmdDictDown)
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
                        }
                    ]
                },
            }
            self.send_action(cmdDictUp)
            sleep(keyboardText["duration_s"])

    def send_keyboard_key_action(self, keyboardKey: Dict[str, Any]) -> None:
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
                    }
                ]
            },
        }
        self.send_action(cmdDictDown)
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
                    }
                ]
            },
        }
        self.send_action(cmdDictUp)
        sleep(keyboardKey["duration_s"])

    def send_mouse_move_action(self, mouseMove: Dict[str, Any]) -> None:
        w: int
        h: int
        w, h = self.get_screen_size()

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
                ]
            },
        }
        self.send_action(cmdDict)
        sleep(mouseMove["duration_s"])

    def send_mouse_click_action(self, mouseClick: Dict[str, Any]) -> None:
        cmdDictDown: Dict[str, Any] = {
            "execute": "input-send-event",
            "arguments": {
                "events": [
                    {
                        "type": "btn",
                        "data": {"down": True, "button": mouseClick["value"]},
                    }
                ]
            },
        }
        self.send_action(cmdDictDown)
        sleep(mouseClick["duration_s"])

        cmdDictUp: Dict[str, Any] = {
            "execute": "input-send-event",
            "arguments": {
                "events": [
                    {
                        "type": "btn",
                        "data": {"down": False, "button": mouseClick["value"]},
                    }
                ]
            },
        }
        self.send_action(cmdDictUp)
        sleep(mouseClick["duration_s"])
