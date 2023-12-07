from typing import Any, List

import cv2
import matplotlib.pyplot as plt


class debugPlot:
    """
    A wrapper class around a matplotlib plot to visualize the current image output detection.
    """

    mseValues: List[float]
    ssimValues: List[float]
    sameImageValues: List[float]
    fig: Any
    axd: Any

    def __init__(self):
        self.mseValues: List[float] = list()
        self.ssimValues: List[float] = list()
        self.sameImageValues: List[float] = list()

        fig, axd = plt.subplot_mosaic(
            [["refImg", "curImg", "difImg"], ["plot", "plot", "plot"]],
            layout="constrained",
        )
        self.fig = fig
        self.axd = axd

    def update_plot(
        self,
        refImg: cv2.typing.MatLike,
        curImg: cv2.typing.MatLike,
        difImage: cv2.typing.MatLike,
        mse: float,
        ssim: float,
        same: float,
    ) -> None:
        """
        Takes the measured and reference image dif and updates the plot accordingly.

        Args:
            refImg (cv2.typing.MatLike): The reference image we are waiting for.
            curImg (cv2.typing.MatLike): The current VM output image.
            difImage (cv2.typing.MatLike): |refImg - curImg| aka the diff of those images.
            mse (float): The mean square error between refImg and curImg.
            ssim (float): The structural similarity index error between refImg and curImg.
            same (float): 1 if refImg and curImg are equal enough and 0 else.
        """
        self.mseValues.append(mse)
        self.ssimValues.append(ssim)
        self.sameImageValues.append(same)

        # Plot the images. Convert images from BGR to RBG.
        self.axd["refImg"].clear()
        self.axd["refImg"].imshow(cv2.cvtColor(refImg, cv2.COLOR_BGR2RGB))
        self.axd["refImg"].set_title("Ref Image")

        self.axd["curImg"].clear()
        self.axd["curImg"].imshow(cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB))
        self.axd["curImg"].set_title("Cur Image")

        self.axd["difImg"].clear()
        self.axd["difImg"].imshow(cv2.cvtColor(difImage, cv2.COLOR_BGR2RGB))
        self.axd["difImg"].set_title("Dif Image")

        # Plot MSE over time
        self.axd["plot"].clear()
        self.axd["plot"].plot(self.mseValues, "bx-", label="MSE over Time")
        self.axd["plot"].plot(self.ssimValues, "rx-", label="SSIM over Time")
        self.axd["plot"].plot(self.sameImageValues, "gx-", label="Same Image")
        self.axd["plot"].set_title("MSE over Time")
        self.axd["plot"].set_xlabel("Iterations")
        self.axd["plot"].set_ylabel("MSE")
        self.axd["plot"].legend()

        plt.pause(0.001)  # Allows the plot to update
