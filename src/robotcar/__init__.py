"""
RobotCar ``test_data`` utilities built on the official SDK models/extrinsics.

Upstream: https://github.com/ori-mrg/robotcar-dataset-sdk
"""

from .calibration import (
    CameraCalibration,
    StereoCalibration,
    load_stereo_calibration,
    pinhole_right_from_left,
    stereo_baseline_m,
)
from .camera_model import CameraModel, PinholeIntrinsics
from .image import load_raw_rgb, load_undistorted_rgb
from .ins import (
    image_timestamp_to_ins_timestamp,
    interpolate_at_image_timestamps,
    interpolate_ins_poses,
)
from .paths import (
    DEFAULT_INS,
    DEFAULT_INS_TIME_OFFSET_US,
    DEFAULT_LEFT_IMAGES,
    DEFAULT_RIGHT_IMAGES,
    MODELS_DIR,
    ROBOTCAR_SDK,
)
from .pose import cam_from_world, world_T_camera
from .stereo import load_stereo_undistorted, sync_stereo_pairs

__all__ = [
    "CameraCalibration",
    "CameraModel",
    "DEFAULT_INS",
    "DEFAULT_INS_TIME_OFFSET_US",
    "DEFAULT_LEFT_IMAGES",
    "DEFAULT_RIGHT_IMAGES",
    "MODELS_DIR",
    "PinholeIntrinsics",
    "ROBOTCAR_SDK",
    "StereoCalibration",
    "cam_from_world",
    "image_timestamp_to_ins_timestamp",
    "interpolate_at_image_timestamps",
    "interpolate_ins_poses",
    "load_raw_rgb",
    "load_stereo_calibration",
    "pinhole_right_from_left",
    "stereo_baseline_m",
    "load_stereo_undistorted",
    "load_undistorted_rgb",
    "sync_stereo_pairs",
    "world_T_camera",
]
