from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
from robotcar import (
    cam_from_world,
    interpolate_at_image_timestamps,
    load_stereo_calibration,
    world_T_camera,
)

from .config import CameraSide, QueryConfig


def infer_query_camera(
    image_path: Path,
    side: CameraSide,
) -> pycolmap.Camera:
    """PINHOLE from undistorted image size + SDK intrinsics (fallback: COLMAP infer)."""
    calib = load_stereo_calibration()
    cam_cal = calib.left if side == "left" else calib.right
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
        ph = cam_cal.pinhole
        return pycolmap.Camera(
            model="PINHOLE",
            width=int(w),
            height=int(h),
            params=[ph.fx, ph.fy, ph.cx, ph.cy],
        )
    except Exception:
        return pycolmap.infer_camera_from_image(image_path)


def ground_truth_cam_from_world(
    timestamp_us: int,
    side: CameraSide,
    config: QueryConfig,
    *,
    origin_timestamp_us: int,
) -> pycolmap.Rigid3d:
    calib = load_stereo_calibration()
    cam_cal = calib.left if side == "left" else calib.right
    poses = interpolate_at_image_timestamps(
        [timestamp_us],
        config.ins_path,
        origin_timestamp=origin_timestamp_us,
        offset_us=config.ins_time_offset_us,
    )
    t_world_cam = world_T_camera(poses[0], cam_cal)
    t_cw = cam_from_world(t_world_cam)
    return pycolmap.Rigid3d(np.asarray(t_cw, dtype=np.float64)[:3, :4])
