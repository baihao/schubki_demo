# Adapted from robotcar-dataset-sdk (CC BY-NC-SA 4.0) — Python 3.

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class PinholeIntrinsics:
    """First line of SDK model .txt (valid for undistorted RGB at native resolution)."""

    fx: float
    fy: float
    cx: float
    cy: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.fx, self.fy, self.cx, self.cy)

    def scaled(self, scale: float) -> PinholeIntrinsics:
        return PinholeIntrinsics(
            self.fx * scale, self.fy * scale, self.cx * scale, self.cy * scale
        )


class CameraModel:
    """RobotCar camera model: intrinsics + bilinear undistortion LUT."""

    def __init__(self, models_dir: Path | str, images_dir: Path | str):
        self.camera: str | None = None
        self.camera_sensor: str | None = None
        self.focal_length: tuple[float, float] | None = None
        self.principal_point: tuple[float, float] | None = None
        self.G_camera_image: np.ndarray | None = None
        self.bilinear_lut: np.ndarray | None = None
        self.model_name: str | None = None
        self._load_intrinsics(Path(models_dir), Path(images_dir))
        self._load_lut(Path(models_dir), Path(images_dir))

    @property
    def pinhole(self) -> PinholeIntrinsics:
        assert self.focal_length and self.principal_point
        return PinholeIntrinsics(
            self.focal_length[0],
            self.focal_length[1],
            self.principal_point[0],
            self.principal_point[1],
        )

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if self.bilinear_lut is None:
            raise RuntimeError("LUT not loaded")
        if image.shape[0] * image.shape[1] != self.bilinear_lut.shape[0]:
            raise ValueError("Incorrect image size for camera model")
        if len(image.shape) == 1:
            raise ValueError("Undistortion requires multi-channel RGB image")
        lut = self.bilinear_lut[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
        undistorted = np.stack(
            [
                map_coordinates(image[:, :, channel], lut, order=1)
                for channel in range(image.shape[2])
            ],
            axis=-1,
        )
        return undistorted.astype(image.dtype)

    def _get_model_name(self, images_dir: Path) -> str:
        images_dir_str = str(images_dir)
        self.camera = re.search(r"(stereo|mono_(left|right|rear))", images_dir_str).group(0)
        if self.camera == "stereo":
            self.camera_sensor = re.search(r"(left|centre|right)", images_dir_str).group(0)
            if self.camera_sensor == "left":
                return "stereo_wide_left"
            if self.camera_sensor == "right":
                return "stereo_wide_right"
            if self.camera_sensor == "centre":
                return "stereo_narrow_left"
            raise RuntimeError("Unknown stereo sensor in path: " + images_dir_str)
        return self.camera

    def _load_intrinsics(self, models_dir: Path, images_dir: Path) -> None:
        self.model_name = self._get_model_name(images_dir)
        intrinsics_path = models_dir / f"{self.model_name}.txt"
        with open(intrinsics_path) as intrinsics_file:
            vals = [float(x) for x in next(intrinsics_file).split()]
            self.focal_length = (vals[0], vals[1])
            self.principal_point = (vals[2], vals[3])
            g_lines = [[float(x) for x in line.split()] for line in intrinsics_file]
            self.G_camera_image = np.array(g_lines, dtype=np.float64)

    def _load_lut(self, models_dir: Path, images_dir: Path) -> None:
        assert self.model_name
        lut_path = models_dir / f"{self.model_name}_distortion_lut.bin"
        lut = np.fromfile(lut_path, np.double)
        lut = lut.reshape([2, lut.size // 2])
        self.bilinear_lut = lut.transpose()
