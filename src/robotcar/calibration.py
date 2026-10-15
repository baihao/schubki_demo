"""Stereo wide left/right intrinsics and extrinsics (official SDK layout)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera_model import CameraModel, PinholeIntrinsics
from .paths import (
    DEFAULT_LEFT_IMAGES,
    DEFAULT_RIGHT_IMAGES,
    EXTRINSICS_DIR,
    MODELS_DIR,
)
from .transform import build_se3_transform


def load_extrinsics_line(path: Path) -> np.ndarray:
    with open(path) as f:
        line = next(f).strip()
    return build_se3_transform([float(x) for x in line.split()])


def camera_to_ins_transform(
    extrinsics_dir: Path,
    images_dir: Path | str,
) -> np.ndarray:
    """
    G_camera_ins: maps points from INS frame into camera native frame.

    Matches SDK ``project_laser_into_camera.py`` when poses come from ``ins.csv``.
    """
    import re

    camera = re.search(r"(stereo|mono_(left|right|rear))", str(images_dir)).group(0)
    g_camera_vehicle = load_extrinsics_line(extrinsics_dir / f"{camera}.txt")
    g_ins_vehicle = load_extrinsics_line(extrinsics_dir / "ins.txt")
    return g_camera_vehicle @ g_ins_vehicle


def stereo_baseline_m(G_camera_image_left: np.ndarray, G_camera_image_right: np.ndarray) -> float:
    """
    Metric distance between left/right pinhole origins (metres).

    RobotCar encodes the horizontal stereo offset in ``models/stereo_wide_*.txt``
    (``G_camera_image``), not in separate per-eye ``extrinsics/`` files.
    Both eyes share ``extrinsics/stereo.txt``; the right model adds ~0.24 m in native frame.
    """
    g_l = np.asarray(G_camera_image_left, dtype=np.float64)
    g_r = np.asarray(G_camera_image_right, dtype=np.float64)
    if g_l.shape != (4, 4) or g_r.shape != (4, 4):
        raise ValueError("G_camera_image must be 4x4")
    # SDK: p_native = G_camera_image @ p_pinhole  →  origin maps to last column translation.
    c_l = g_l[0:3, 3]
    c_r = g_r[0:3, 3]
    return float(np.linalg.norm(c_r - c_l))


def pinhole_right_from_left(G_camera_image_left: np.ndarray, G_camera_image_right: np.ndarray) -> np.ndarray:
    """4×4 ``T`` with ``p_pinhole_right = T @ p_pinhole_left`` (homogeneous)."""
    g_l = np.asarray(G_camera_image_left, dtype=np.float64)
    g_r = np.asarray(G_camera_image_right, dtype=np.float64)
    return np.linalg.inv(g_r) @ g_l


@dataclass(frozen=True)
class CameraCalibration:
    side: str
    images_dir: Path
    model: CameraModel
    G_camera_ins: np.ndarray
    G_camera_image: np.ndarray

    @property
    def pinhole(self) -> PinholeIntrinsics:
        return self.model.pinhole


@dataclass(frozen=True)
class StereoCalibration:
    left: CameraCalibration
    right: CameraCalibration

    @property
    def baseline_m(self) -> float:
        """Left–right pinhole baseline (m) from SDK ``G_camera_image`` pair."""
        return stereo_baseline_m(self.left.G_camera_image, self.right.G_camera_image)

    @property
    def T_pinhole_right_from_left(self) -> np.ndarray:
        """Rigid transform from left to right pinhole frame at the same timestamp."""
        return pinhole_right_from_left(self.left.G_camera_image, self.right.G_camera_image)


def load_stereo_calibration(
    models_dir: Path | None = None,
    extrinsics_dir: Path | None = None,
    left_images_dir: Path | None = None,
    right_images_dir: Path | None = None,
) -> StereoCalibration:
    """
    Load wide stereo left/right intrinsics (model + LUT) and extrinsics to INS.

    Requires checkout of https://github.com/ori-mrg/robotcar-dataset-sdk
    (``models/``, ``extrinsics/``). Run ``scripts/setup_robotcar_sdk.sh`` if missing.
    """
    models_dir = Path(models_dir or MODELS_DIR)
    extrinsics_dir = Path(extrinsics_dir or EXTRINSICS_DIR)
    left_images_dir = Path(left_images_dir or DEFAULT_LEFT_IMAGES)
    right_images_dir = Path(right_images_dir or DEFAULT_RIGHT_IMAGES)

    if not (models_dir / "stereo_wide_left.txt").is_file():
        raise FileNotFoundError(
            f"Missing SDK models under {models_dir}. "
            "Clone robotcar-dataset-sdk or set ROBOTCAR_SDK."
        )

    left_model = CameraModel(models_dir, left_images_dir)
    right_model = CameraModel(models_dir, right_images_dir)

    left = CameraCalibration(
        side="left",
        images_dir=left_images_dir,
        model=left_model,
        G_camera_ins=camera_to_ins_transform(extrinsics_dir, left_images_dir),
        G_camera_image=left_model.G_camera_image.copy(),
    )
    right = CameraCalibration(
        side="right",
        images_dir=right_images_dir,
        model=right_model,
        G_camera_ins=camera_to_ins_transform(extrinsics_dir, right_images_dir),
        G_camera_image=right_model.G_camera_image.copy(),
    )
    return StereoCalibration(left=left, right=right)
