"""World / camera transforms for INS-driven mapping (COLMAP ``cam_from_world`` helpers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .calibration import CameraCalibration


def world_T_camera(
    T_world_ins: np.ndarray,
    calib: CameraCalibration,
) -> np.ndarray:
    """
    4×4 ``T_world←cam_pinhole``: maps OpenCV pinhole points (x right, y down, z forward) to world.

    ``T_world_ins`` from :func:`robotcar.ins.interpolate_ins_poses`.
    """
    g_cam_ins = calib.G_camera_ins
    g_cam_image = calib.G_camera_image
    return T_world_ins @ np.linalg.inv(g_cam_ins) @ g_cam_image


def cam_from_world(T_world_cam: np.ndarray) -> np.ndarray:
    """COLMAP naming: camera-from-world = inverse of world-from-camera."""
    return np.linalg.inv(T_world_cam)
