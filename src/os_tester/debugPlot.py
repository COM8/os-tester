from typing import List, Any
import matplotlib.pyplot as plt
import cv2


class debugPlot:
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
