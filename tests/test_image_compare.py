import os

import cv2
import numpy as np
import pytest

try:
    import libvirt  # noqa: F401
except Exception:
    pytest.skip("libvirt is required to import os_tester.vm", allow_module_level=True)

from os_tester.vm import vm


def _load_image(file_path: str) -> np.ndarray:
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to load image from '{file_path}'")
    return image


def _compare_images(img_a: np.ndarray, img_b: np.ndarray) -> tuple[float, float]:
    tester = vm(None, "pytest")
    mse, ssim, _ = tester._vm__comp_images(img_a, img_b)
    return mse, ssim


def _compare_images_test(file_name_a: str, file_name_b: str, mse_expected: float, ssim_expected: float):
    basePath: str = f"{os.path.dirname(os.path.abspath(__file__))}/images"
    img_a = _load_image(f"{basePath}/{file_name_a}")
    img_b = _load_image(f"{basePath}/{file_name_b}")
    mse, ssim = _compare_images(img_a, img_b)

    assert mse == pytest.approx(mse_expected, abs=1e-3)
    assert ssim == pytest.approx(ssim_expected, abs=1e-3)


def test_compare_same_image_file() -> None:
    _compare_images_test("a.png", "a.png", 0.0, 1.0)


def test_compare_similar_image() -> None:
    _compare_images_test("a.png", "b.png", 0.0, 1.0)
    _compare_images_test("a.png", "c.png", 6.005, 0.0)  # The difference is the cursor


def test_compare_images_identical() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mse, ssim = _compare_images(img, img.copy())

    assert mse == pytest.approx(0.0, abs=1e-6)
    assert ssim == pytest.approx(1.0, abs=1e-6)
