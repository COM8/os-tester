import cv2
import numpy as np
import pytest

from os_tester.stages import area, stages


def _write_stage_file(tmp_path, stage_dict) -> None:
    stage_path = tmp_path / "stages.yml"
    stage_path.write_text(stage_dict, encoding="utf-8")


def _write_ref_image(tmp_path, name: str = "ref.png") -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / name), img)


def test_stages_parsing_accepts_area(tmp_path) -> None:
    _write_ref_image(tmp_path)
    stage_yaml = """
stages:
  - stage: "boot"
    timeout_s: 5
    paths:
      - path:
          checks:
            - path: "ref.png"
              ssim_geq: 0.9
              area:
                x1Percentage: 0.1
                x2Percentage: 0.9
                y1Percentage: 0.2
                y2Percentage: 0.8
          actions: []
          nextStage: "done"
"""
    _write_stage_file(tmp_path, stage_yaml)

    loaded = stages(str(tmp_path), "stages")
    imageArea = loaded.stagesList[0].pathsList[0].checkList[0].area

    assert isinstance(imageArea, area)
    assert imageArea.x1Percentage == pytest.approx(0.1)


def test_stages_parsing_rejects_invalid_area(tmp_path) -> None:
    _write_ref_image(tmp_path)
    stage_yaml = """
stages:
  - stage: "boot"
    timeout_s: 5
    paths:
      - path:
          checks:
            - path: "ref.png"
              ssim_geq: 0.9
              area:
                x1Percentage: 0.9
                x2Percentage: 0.1
                y1Percentage: 0.2
                y2Percentage: 0.8
          actions: []
          nextStage: "done"
"""
    _write_stage_file(tmp_path, stage_yaml)

    with pytest.raises(ValueError, match="x1Percentage < x2Percentage"):
        stages(str(tmp_path), "stages")


def test_stages_parsing_rejects_invalid_ssim(tmp_path) -> None:
    _write_ref_image(tmp_path)
    stage_yaml = """
stages:
  - stage: "boot"
    timeout_s: 5
    paths:
      - path:
          checks:
            - path: "ref.png"
              ssim_geq: 1.5
          actions: []
          nextStage: "done"
"""
    _write_stage_file(tmp_path, stage_yaml)

    with pytest.raises(ValueError, match="ssim_geq"):
        stages(str(tmp_path), "stages")
