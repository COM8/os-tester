import os

import cv2
import numpy as np
import pytest
from typing import Optional

try:
    import libvirt  # noqa: F401
except Exception:
    pytest.skip("libvirt is required to import os_tester.vm", allow_module_level=True)

from os_tester.vm import vm
from os_tester.stages import area


def _load_image(file_path: str) -> np.ndarray:
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Failed to load image from '{file_path}'")
    return image


def _compare_images(img_a: np.ndarray, img_b: np.ndarray, imageArea: Optional[area] = None) -> float:
    tester = vm(None, "pytest")
    ssim, _ = tester._vm__comp_images(img_a, img_b, imageArea)
    return ssim


def _compare_images_test(file_name_a: str, file_name_b: str, ssim_expected: float):
    basePath: str = f"{os.path.dirname(os.path.abspath(__file__))}/images"
    img_a = _load_image(f"{basePath}/{file_name_a}")
    img_b = _load_image(f"{basePath}/{file_name_b}")
    ssim = _compare_images(img_a, img_b)

    assert ssim == pytest.approx(ssim_expected, abs=1e-2)


def test_compare_same_image_file() -> None:
    _compare_images_test("a.png", "a.png", 1.0)


def test_compare_similar_image() -> None:
    _compare_images_test("a.png", "b.png", 1.0)
    _compare_images_test("a.png", "c.png", 0.99)  # The difference is the cursor


def test_compare_similar_luks_image() -> None:
    _compare_images_test("luks_a.png", "luks_b.png", 0.97)


def test_compare_images_identical() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ssim = _compare_images(img, img.copy())

    assert ssim == pytest.approx(1.0, abs=1e-2)


def test_compare_images_with_area_excludes_difference() -> None:
    img_a = np.zeros((100, 100, 3), dtype=np.uint8)
    img_b = img_a.copy()
    img_b[0, 0] = [255, 255, 255]

    imageArea = area(
        {
            "x1Percentage": 0.5,
            "x2Percentage": 1.0,
            "y1Percentage": 0.5,
            "y2Percentage": 1.0,
        }
    )
    ssim = _compare_images(img_a, img_b, imageArea)
    assert ssim == pytest.approx(1.0, abs=1e-2)


def test_compare_images_with_area_includes_difference() -> None:
    img_a = np.zeros((10, 10, 3), dtype=np.uint8)
    img_b = img_a.copy()
    img_b[0, 0] = [255, 255, 255]

    imageArea = area(
        {
            "x1Percentage": 0.0,
            "x2Percentage": 0.9,
            "y1Percentage": 0.0,
            "y2Percentage": 0.9,
        }
    )
    ssim = _compare_images(img_a, img_b, imageArea)
    assert ssim < 0.999
